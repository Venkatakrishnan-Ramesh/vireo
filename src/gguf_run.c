/* vireo-gguf: run a real GGUF model (Qwen2 family, Q8_0) on CPU with weights
 * mmap'd from disk and the KV cache streamed through vireo's out-of-core store.
 * Tokenizer: GPT-2 style byte-level BPE read from the GGUF metadata.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gguf.h"
#include "gguf_compute.h"
#include "kvstore.h"
#include "expert_store.h"
#include "select/select.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static double wall(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

/* ----------------------------- byte <-> unicode (GPT-2) ------------------ */
static int BYTE2CP[256];
static int CP2BYTE[512];   /* codepoints used are < 512 */
static void build_byte_maps(void) {
    int used[256] = {0}, cp = 256, n = 0;
    for (int b = '!'; b <= '~'; b++) used[b] = 1;
    for (int b = 0xA1; b <= 0xAC; b++) used[b] = 1;
    for (int b = 0xAE; b <= 0xFF; b++) used[b] = 1;
    for (int b = 0; b < 256; b++) {
        if (used[b]) BYTE2CP[b] = b;
        else BYTE2CP[b] = cp + n++;
    }
    for (int i = 0; i < 512; i++) CP2BYTE[i] = -1;
    for (int b = 0; b < 256; b++) CP2BYTE[BYTE2CP[b]] = b;
}
/* encode a UTF-8 codepoint to UTF-8 bytes into buf, return length */
static int cp_to_utf8(int cp, char *buf) {
    if (cp < 0x80) { buf[0]=cp; return 1; }
    if (cp < 0x800) { buf[0]=0xC0|(cp>>6); buf[1]=0x80|(cp&0x3F); return 2; }
    buf[0]=0xE0|(cp>>12); buf[1]=0x80|((cp>>6)&0x3F); buf[2]=0x80|(cp&0x3F); return 3;
}
/* decode one UTF-8 codepoint from s, advance *i */
static int utf8_next(const char *s, int len, int *i) {
    unsigned char c = s[*i];
    if (c < 0x80) { (*i)++; return c; }
    if ((c>>5)==0x6 && *i+1<len) { int cp=((c&0x1F)<<6)|(s[*i+1]&0x3F); *i+=2; return cp; }
    if ((c>>4)==0xE && *i+2<len) { int cp=((c&0xF)<<12)|((s[*i+1]&0x3F)<<6)|(s[*i+2]&0x3F); *i+=3; return cp; }
    (*i)++; return c;
}

/* ----------------------------- tokenizer -------------------------------- */
typedef struct { gguf_str *tok; int n; int *merge_rank_table; } Vocab;

/* materialize a string array from gguf metadata into a gguf_str* */
static gguf_str *materialize_str_array(const gguf_kv *kv, uint64_t *out_n) {
    gguf_str *a = malloc(kv->arr_len * sizeof(gguf_str));
    const uint8_t *p = kv->arr_data;
    for (uint64_t i = 0; i < kv->arr_len; i++) {
        uint64_t n; memcpy(&n, p, 8); p += 8;
        a[i].ptr = (const char *)p; a[i].len = n; p += n;
    }
    *out_n = kv->arr_len; return a;
}

/* simple open-addressing hash map: token string -> id */
typedef struct { gguf_str *keys; int *ids; int cap; } Map;
static uint64_t hash_bytes(const char *s, uint64_t n){ uint64_t h=1469598103934665603ULL; for(uint64_t i=0;i<n;i++){h^=(unsigned char)s[i];h*=1099511628211ULL;} return h; }
static Map map_build(gguf_str *toks, int n) {
    Map m; m.cap = 1; while (m.cap < n*2) m.cap <<= 1;
    m.keys = calloc(m.cap, sizeof(gguf_str)); m.ids = malloc(m.cap*sizeof(int));
    for (int i=0;i<m.cap;i++) m.ids[i]=-1;
    for (int i=0;i<n;i++){ uint64_t h=hash_bytes(toks[i].ptr,toks[i].len)&(m.cap-1);
        while (m.ids[h]!=-1) h=(h+1)&(m.cap-1);
        m.keys[h]=toks[i]; m.ids[h]=i; }
    return m;
}
static int map_get(const Map *m, const char *s, uint64_t n) {
    uint64_t h=hash_bytes(s,n)&(m->cap-1);
    while (m->ids[h]!=-1) { if (m->keys[h].len==n && memcmp(m->keys[h].ptr,s,n)==0) return m->ids[h]; h=(h+1)&(m->cap-1); }
    return -1;
}

/* merges: "left right" -> rank. Store in a hash map keyed by "left\x1Fright". */
typedef struct { char **key; int *rank; int cap; } Merges;
static Merges merges_build(gguf_str *mg, int n) {
    Merges M; M.cap=1; while(M.cap<n*2) M.cap<<=1;
    M.key=calloc(M.cap,sizeof(char*)); M.rank=malloc(M.cap*sizeof(int)); for(int i=0;i<M.cap;i++)M.rank[i]=-1;
    for (int i=0;i<n;i++){
        char buf[512]; int L=mg[i].len<500?mg[i].len:500; memcpy(buf,mg[i].ptr,L); buf[L]=0;
        char *sp=strchr(buf,' '); if(!sp) continue; *sp=0x1F;
        uint64_t h=hash_bytes(buf,L)&(M.cap-1);
        while(M.key[h]){ if(!memcmp(M.key[h],buf,L+1)){break;} h=(h+1)&(M.cap-1);} 
        if(!M.key[h]){ M.key[h]=malloc(L+1); memcpy(M.key[h],buf,L+1); M.rank[h]=i; }
    }
    return M;
}
static int merge_rank(const Merges *M, const char *a, int al, const char *b, int bl) {
    char buf[512]; if (al+bl+1>510) return -1;
    memcpy(buf,a,al); buf[al]=0x1F; memcpy(buf+al+1,b,bl); int L=al+1+bl; buf[L]=0;
    uint64_t h=hash_bytes(buf,L)&(M->cap-1);
    while(M->key[h]){ if(!memcmp(M->key[h],buf,L+1)) return M->rank[h]; h=(h+1)&(M->cap-1);} 
    return -1;
}

/* encode UTF-8 text -> token ids (byte-level BPE, split before each space-marker) */
static int encode(const char *text, const Map *vmap, const Merges *M, int *out, int max_out) {
    /* byte-encode to a codepoint string (as UTF-8) */
    int tl = strlen(text);
    char *enc = malloc((size_t)tl*3 + 16); int el = 0;
    for (int i=0;i<tl;i++) el += cp_to_utf8(BYTE2CP[(unsigned char)text[i]], enc+el);
    /* split into words before each 'Ġ' (U+0120) */
    int nout = 0;
    int i = 0;
    while (i < el) {
        int wstart = i;
        int cp = utf8_next(enc, el, &i);  /* first cp of word */
        (void)cp;
        while (i < el) { int j=i; int c=utf8_next(enc,el,&j); if (c==0x120) break; i=j; }
        /* word = enc[wstart..i); split into symbol list of codepoint-strings */
        char *syms[1024]; int sl[1024], ns=0;
        int k=wstart;
        while (k<i) { int s=k; utf8_next(enc,el,&k); syms[ns]=enc+s; sl[ns]=k-s; ns++; }
        /* BPE merge loop */
        for (;;) {
            int best=-1, bi=-1;
            for (int a=0;a<ns-1;a++){ int r=merge_rank(M,syms[a],sl[a],syms[a+1],sl[a+1]); if(r!=-1 && (best==-1||r<best)){best=r;bi=a;} }
            if (bi<0) break;
            /* merge bi and bi+1 (contiguous in enc buffer, so just extend length) */
            sl[bi]+=sl[bi+1];
            for (int a=bi+1;a<ns-1;a++){ syms[a]=syms[a+1]; sl[a]=sl[a+1]; }
            ns--;
        }
        for (int a=0;a<ns;a++){ int id=map_get(vmap,syms[a],sl[a]); if(id>=0 && nout<max_out) out[nout++]=id; }
    }
    free(enc);
    return nout;
}

/* ----------------------------- model ------------------------------------ */
typedef struct {
    const gguf_tensor *attn_norm,*wq,*bq,*wk,*bk,*wv,*bv,*wo,*ffn_norm,*wg,*wu,*wd;
    const gguf_tensor *q_norm,*k_norm;                 /* optional QK-norm (OLMoE/Qwen3) */
    const gguf_tensor *router,*gate_exps,*up_exps,*down_exps; /* MoE */
} Layer;
typedef struct {
    gguf_ctx *g;
    int n_layers,n_heads,n_kv_heads,head_size,dim,ffn,vocab;
    float eps, rope_base;
    const gguf_tensor *tok_embd,*out_norm,*lm_head;
    Layer *L;
    KVStore *kv;
    float *x,*xb,*h1,*h2,*q,*k,*v,*att_out,*logits;
    int is_moe,n_experts,n_used,ffn_exp; uint32_t exp_type;
    ExpertStore *es;
    float *rlog,*gbuf,*ubuf,*dbuf,*moe_acc;
} Model;

static const gguf_tensor *T(gguf_ctx*g,const char*fmt,int l){char b[128];snprintf(b,sizeof(b),fmt,l);const gguf_tensor*t=gguf_tensor_by_name(g,b);if(!t){fprintf(stderr,"missing tensor %s\n",b);exit(1);}return t;}

static void rmsnorm(float*o,const float*x,const gguf_tensor*w,int n,float eps){
    float ss=0; for(int i=0;i<n;i++)ss+=x[i]*x[i]; float r=1.0f/sqrtf(ss/n+eps);
    float wrow[16384]; gguf_dequant_row(w->ggml_type,w->data,wrow,n);
    for(int i=0;i<n;i++)o[i]=x[i]*r*wrow[i];
}
static void add_bias(float*y,const gguf_tensor*b,int n){ if(!b)return; float br[16384]; gguf_dequant_row(b->ggml_type,b->data,br,n); for(int i=0;i<n;i++)y[i]+=br[i]; }
/* NEOX RoPE: rotate (j, j+hs/2) pairs, per head */
static void rope_neox(float*vec,int n_heads,int hs,int pos,float base){
    int half=hs/2;
    for(int h=0;h<n_heads;h++){ float*p=vec+h*hs;
        for(int j=0;j<half;j++){ float freq=powf(base,-2.0f*j/hs); float a=pos*freq; float c=cosf(a),s=sinf(a);
            float x0=p[j],x1=p[j+half]; p[j]=x0*c-x1*s; p[j+half]=x0*s+x1*c; } }
}


/* softmax over n in place */
static void softmax_n(float*v,int n){ float m=v[0]; for(int i=1;i<n;i++)if(v[i]>m)m=v[i];
    float s=0; for(int i=0;i<n;i++){v[i]=expf(v[i]-m); s+=v[i];} for(int i=0;i<n;i++)v[i]/=s; }

/* run the MoE FFN for the current token: router -> top-k experts via ExpertStore */
static void moe_ffn(Model*m,int layer,float*hb){
    Layer*ly=&m->L[layer]; int d=m->dim, fe=m->ffn_exp, ne=m->n_experts, k=m->n_used;
    gguf_matmul(m->rlog,hb,ly->router,d,ne);
    softmax_n(m->rlog,ne);
    /* pick top-k by probability */
    int idxs[64]; float wts[64]; int kk = k<64?k:64;
    for(int t=0;t<kk;t++){ int best=-1; float bv=-1; for(int e=0;e<ne;e++){ int used=0; for(int j=0;j<t;j++) if(idxs[j]==e){used=1;break;} if(!used && m->rlog[e]>bv){bv=m->rlog[e];best=e;} } idxs[t]=best; wts[t]=bv; }
    for(int i=0;i<d;i++) m->moe_acc[i]=0;
    uint64_t base_g=(uint64_t)((const uint8_t*)ly->gate_exps->data-(const uint8_t*)m->g->map);
    uint64_t base_u=(uint64_t)((const uint8_t*)ly->up_exps->data-(const uint8_t*)m->g->map);
    uint64_t base_d=(uint64_t)((const uint8_t*)ly->down_exps->data-(const uint8_t*)m->g->map);
    uint64_t gslice=ly->gate_exps->nbytes/ne, uslice=ly->up_exps->nbytes/ne, dslice=ly->down_exps->nbytes/ne;
    for(int t=0;t<kk;t++){
        int e=idxs[t]; float w=wts[t];   /* OLMoE: raw softmax prob, no renorm */
        ExpertView v;
        if(expert_store_lookup(m->es,layer,e,&v)) continue;
        gguf_tensor tg={0}; tg.data=v.gate; tg.ggml_type=ly->gate_exps->ggml_type;
        gguf_tensor tu={0}; tu.data=v.up;   tu.ggml_type=ly->up_exps->ggml_type;
        gguf_tensor td={0}; td.data=v.down; td.ggml_type=ly->down_exps->ggml_type;
        gguf_matmul(m->gbuf,hb,&tg,d,fe);
        gguf_matmul(m->ubuf,hb,&tu,d,fe);
        for(int i=0;i<fe;i++){ float g=m->gbuf[i]; m->gbuf[i]=(g/(1.0f+expf(-g)))*m->ubuf[i]; }
        gguf_matmul(m->dbuf,m->gbuf,&td,fe,d);
        for(int i=0;i<d;i++) m->moe_acc[i]+=w*m->dbuf[i];
        expert_store_release(m->es,&v);
    }
    (void)base_g;(void)base_u;(void)base_d;(void)gslice;(void)uslice;(void)dslice;
    for(int i=0;i<d;i++) m->x[i]+=m->moe_acc[i];
}

static float* forward(Model*m,int token,int pos,int want_logits){
    int d=m->dim,hs=m->head_size;
    gguf_dequant_row(m->tok_embd->ggml_type, (const char*)m->tok_embd->data + (size_t)token*(m->tok_embd->nbytes/m->vocab), m->x, d);
    for(int l=0;l<m->n_layers;l++){
        Layer*ly=&m->L[l];
        rmsnorm(m->xb,m->x,ly->attn_norm,d,m->eps);
        gguf_matmul(m->q,m->xb,ly->wq,d,m->n_heads*hs); add_bias(m->q,ly->bq,m->n_heads*hs);
        gguf_matmul(m->k,m->xb,ly->wk,d,m->n_kv_heads*hs); add_bias(m->k,ly->bk,m->n_kv_heads*hs);
        gguf_matmul(m->v,m->xb,ly->wv,d,m->n_kv_heads*hs); add_bias(m->v,ly->bv,m->n_kv_heads*hs);
        if(ly->q_norm) rmsnorm(m->q,m->q,ly->q_norm,m->n_heads*hs,m->eps);
        if(ly->k_norm) rmsnorm(m->k,m->k,ly->k_norm,m->n_kv_heads*hs,m->eps);
        rope_neox(m->q,m->n_heads,hs,pos,m->rope_base);
        rope_neox(m->k,m->n_kv_heads,hs,pos,m->rope_base);
        for(int kh=0;kh<m->n_kv_heads;kh++)
            kv_append(m->kv, l*m->n_kv_heads+kh, pos, m->k+kh*hs, m->v+kh*hs);
        int kv_mul=m->n_heads/m->n_kv_heads;
        for(int hh=0;hh<m->n_heads;hh++)
            kv_attend(m->kv, l*m->n_kv_heads+(hh/kv_mul), m->q+hh*hs, m->att_out+hh*hs);
        gguf_matmul(m->xb,m->att_out,ly->wo,m->n_heads*hs,d);
        for(int i=0;i<d;i++)m->x[i]+=m->xb[i];
        rmsnorm(m->xb,m->x,ly->ffn_norm,d,m->eps);
        if(m->is_moe){ moe_ffn(m,l,m->xb); }
        else {
            gguf_matmul(m->h1,m->xb,ly->wg,d,m->ffn);
            gguf_matmul(m->h2,m->xb,ly->wu,d,m->ffn);
            for(int i=0;i<m->ffn;i++){ float g=m->h1[i]; m->h1[i]=(g/(1.0f+expf(-g)))*m->h2[i]; }
            gguf_matmul(m->xb,m->h1,ly->wd,m->ffn,d);
            for(int i=0;i<d;i++)m->x[i]+=m->xb[i];
        }
    }
    if(!want_logits) return NULL;              /* prefill: skip the 138MB LM head */
    rmsnorm(m->xb,m->x,m->out_norm,d,m->eps);
    gguf_matmul(m->logits,m->xb,m->lm_head,d,m->vocab);
    return m->logits;
}

static int argmax(const float*v,int n){int bi=0;float bv=v[0];for(int i=1;i<n;i++)if(v[i]>bv){bv=v[i];bi=i;}return bi;}

int main(int argc,char**argv){
    const char*path=NULL,*prompt="The meaning of life is",*pfile=NULL;
    int steps=80;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-p")&&i+1<argc)prompt=argv[++i];
        else if(!strcmp(argv[i],"-n")&&i+1<argc)steps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-f")&&i+1<argc)pfile=argv[++i];
        else if(argv[i][0]!='-')path=argv[i];
    }
    if(!path){fprintf(stderr,"usage: %s model.gguf [-p prompt] [-n steps]\n",argv[0]);return 2;}
#ifdef _OPENMP
    if (!getenv("OMP_NUM_THREADS")) { int nc = omp_get_num_procs(); omp_set_num_threads(nc < 4 ? nc : 4); }
#endif
    build_byte_maps();
    gguf_ctx*g=gguf_open(path); if(!g)return 1;
    char *pbuf=NULL;
    if(pfile){ FILE*fp=fopen(pfile,"rb"); if(!fp){fprintf(stderr,"cannot open %s\n",pfile);return 1;}
        fseek(fp,0,SEEK_END); long sz=ftell(fp); fseek(fp,0,SEEK_SET);
        pbuf=malloc(sz+1); if(fread(pbuf,1,sz,fp)!=(size_t)sz){} pbuf[sz]=0; fclose(fp); prompt=pbuf; }
    Model m={0}; m.g=g;
    uint64_t u; char arch[64]="qwen2"; { uint64_t al; const char*a=gguf_get_str(g,"general.architecture",&al); if(a){ int n=al<63?al:63; memcpy(arch,a,n); arch[n]=0; } }
    char kb[128];
    #define GK(suf) (snprintf(kb,sizeof(kb),"%s." suf,arch),kb)
    gguf_get_u64(g,GK("block_count"),&u); m.n_layers=u;
    gguf_get_u64(g,GK("embedding_length"),&u); m.dim=u;
    gguf_get_u64(g,GK("feed_forward_length"),&u); m.ffn=u;
    gguf_get_u64(g,GK("attention.head_count"),&u); m.n_heads=u;
    gguf_get_u64(g,GK("attention.head_count_kv"),&u); m.n_kv_heads=u;
    gguf_get_f32(g,GK("attention.layer_norm_rms_epsilon"),&m.eps);
    m.rope_base=10000.0f; gguf_get_f32(g,GK("rope.freq_base"),&m.rope_base);
    m.head_size=m.dim/m.n_heads;
    /* MoE fields (present only for MoE architectures) */
    if(gguf_get_u64(g,GK("expert_count"),&u)==0 && u>0){ m.is_moe=1; m.n_experts=u;
        m.n_used=8; if(gguf_get_u64(g,GK("expert_used_count"),&u)==0) m.n_used=u;
        m.ffn_exp=m.ffn; if(gguf_get_u64(g,GK("expert_feed_forward_length"),&u)==0) m.ffn_exp=u; }
    m.tok_embd=gguf_tensor_by_name(g,"token_embd.weight");
    m.vocab=m.tok_embd->dims[1];
    m.out_norm=gguf_tensor_by_name(g,"output_norm.weight");
    m.lm_head=gguf_tensor_by_name(g,"output.weight"); if(!m.lm_head)m.lm_head=m.tok_embd;
    if(!m.out_norm){fprintf(stderr,"no output_norm.weight\n");return 1;}
    m.L=calloc(m.n_layers,sizeof(Layer));
    for(int l=0;l<m.n_layers;l++){Layer*ly=&m.L[l];
        ly->attn_norm=T(g,"blk.%d.attn_norm.weight",l);
        ly->wq=T(g,"blk.%d.attn_q.weight",l); ly->bq=gguf_tensor_by_name(g,(char[64]){0}); 
        char nb[64];
        snprintf(nb,64,"blk.%d.attn_q.bias",l); ly->bq=gguf_tensor_by_name(g,nb);
        ly->wk=T(g,"blk.%d.attn_k.weight",l); snprintf(nb,64,"blk.%d.attn_k.bias",l); ly->bk=gguf_tensor_by_name(g,nb);
        ly->wv=T(g,"blk.%d.attn_v.weight",l); snprintf(nb,64,"blk.%d.attn_v.bias",l); ly->bv=gguf_tensor_by_name(g,nb);
        ly->wo=T(g,"blk.%d.attn_output.weight",l);
        ly->ffn_norm=T(g,"blk.%d.ffn_norm.weight",l);
        snprintf(nb,64,"blk.%d.attn_q_norm.weight",l); ly->q_norm=gguf_tensor_by_name(g,nb);
        snprintf(nb,64,"blk.%d.attn_k_norm.weight",l); ly->k_norm=gguf_tensor_by_name(g,nb);
        if(m.is_moe){
            ly->router   =T(g,"blk.%d.ffn_gate_inp.weight",l);
            ly->gate_exps=T(g,"blk.%d.ffn_gate_exps.weight",l);
            ly->up_exps  =T(g,"blk.%d.ffn_up_exps.weight",l);
            ly->down_exps=T(g,"blk.%d.ffn_down_exps.weight",l);
        } else {
            ly->wg=T(g,"blk.%d.ffn_gate.weight",l); ly->wu=T(g,"blk.%d.ffn_up.weight",l); ly->wd=T(g,"blk.%d.ffn_down.weight",l);
        }
    }
    /* ExpertStore: stream MoE expert weights from disk */
    if(m.is_moe){
        m.exp_type=m.L[0].gate_exps->ggml_type;
        uint64_t ecap_mb = getenv("VIREO_EXPERT_CAP_MB")?strtoull(getenv("VIREO_EXPERT_CAP_MB"),0,10):0;
        m.es=expert_store_create(g->fd, ecap_mb*1048576ULL, m.n_layers, m.n_experts);
        for(int l=0;l<m.n_layers;l++){ Layer*ly=&m.L[l];
            uint64_t bg=(uint64_t)((const uint8_t*)ly->gate_exps->data-(const uint8_t*)g->map);
            uint64_t bu=(uint64_t)((const uint8_t*)ly->up_exps->data-(const uint8_t*)g->map);
            uint64_t bd=(uint64_t)((const uint8_t*)ly->down_exps->data-(const uint8_t*)g->map);
            uint32_t gs=ly->gate_exps->nbytes/m.n_experts, us=ly->up_exps->nbytes/m.n_experts, ds=ly->down_exps->nbytes/m.n_experts;
            for(int e=0;e<m.n_experts;e++){ ExpertSlices L2={ bg+(uint64_t)e*gs, bu+(uint64_t)e*us, bd+(uint64_t)e*ds, gs,us,ds };
                expert_store_register(m.es,l,e,&L2); }
        }
    }
    /* vireo KV store */
    int capb = getenv("VIREO_CAP_BLOCKS")?atoi(getenv("VIREO_CAP_BLOCKS")):0;
    KVConfig cfg={0}; cfg.n_layers=m.n_layers*m.n_kv_heads; cfg.kv_dim=m.head_size;
    cfg.block_tokens = getenv("VIREO_BLOCK")?atoi(getenv("VIREO_BLOCK")):32;
    cfg.sink_tokens   = getenv("VIREO_SINK")?atoi(getenv("VIREO_SINK")):32;
    cfg.window_tokens = getenv("VIREO_WINDOW")?atoi(getenv("VIREO_WINDOW")):256;
    uint64_t bb=(uint64_t)cfg.block_tokens*m.head_size*2*sizeof(float);
    cfg.capacity_bytes=capb>0?(uint64_t)capb*bb:0;
    const char*sn=getenv("VIREO_STRATEGY"); const SelectStrategy*strat=&VIREO_SELECT_ALL;
    if(sn&&!strcmp(sn,"recent"))strat=&VIREO_SELECT_RECENT; else if(sn&&!strcmp(sn,"topk"))strat=vireo_select_topk(getenv("VIREO_TOPK")?atoi(getenv("VIREO_TOPK")):16);
    m.kv=kvstore_ooc_create(cfg,strat,getenv("VIREO_WORKDIR"));

    m.x=malloc(m.dim*4);m.xb=malloc(m.dim*4);m.h1=malloc(m.ffn*4);m.h2=malloc(m.ffn*4);
    m.q=malloc(m.n_heads*m.head_size*4);m.k=malloc(m.n_kv_heads*m.head_size*4);m.v=malloc(m.n_kv_heads*m.head_size*4);
    m.att_out=malloc(m.n_heads*m.head_size*4);m.logits=malloc(m.vocab*4);
    if(m.is_moe){ m.rlog=malloc(m.n_experts*4); m.gbuf=malloc(m.ffn_exp*4); m.ubuf=malloc(m.ffn_exp*4);
        m.dbuf=malloc(m.dim*4); m.moe_acc=malloc(m.dim*4); }

    /* tokenizer */
    uint64_t nt,nm; const gguf_kv*tk=gguf_find(g,"tokenizer.ggml.tokens"); const gguf_kv*mk=gguf_find(g,"tokenizer.ggml.merges");
    gguf_str*toks=materialize_str_array(tk,&nt); gguf_str*mgs=materialize_str_array(mk,&nm);
    Map vmap=map_build(toks,nt); Merges M=merges_build(mgs,nm);
    uint64_t eos=151645; gguf_get_u64(g,"tokenizer.ggml.eos_token_id",&eos);

    int maxids=300000; int *ids=malloc(sizeof(int)*maxids); int np=encode(prompt,&vmap,&M,ids,maxids);
    if(m.is_moe) fprintf(stderr,"[vireo-gguf] %s | arch=%s MoE %d experts, top-%d, ffn_exp=%d | layers=%d heads=%d/%d | prompt=%d tokens\n",
            path,arch,m.n_experts,m.n_used,m.ffn_exp,m.n_layers,m.n_heads,m.n_kv_heads,np);
    else fprintf(stderr,"[vireo-gguf] %s | arch=%s layers=%d heads=%d/%d hs=%d vocab=%d | prompt=%d tokens | strategy=%s cap_blocks=%d\n",
            path,arch,m.n_layers,m.n_heads,m.n_kv_heads,m.head_size,m.vocab,np,strat->name,capb);

    if(np<200) printf("%s",prompt); else printf("[prompt: %d tokens]\n",np); fflush(stdout);
    double t0=wall();
    int pos=0, tok=ids[0];
    for(int step=0; step<steps; step++){
        float*lg=forward(&m,tok,pos, pos>=np-1); pos++;
        int next;
        if(pos<np) next=ids[pos];                 /* still feeding prompt */
        else next=argmax(lg,m.vocab);             /* greedy generate */
        if(pos>=np){
            /* decode + print next */
            gguf_str s=toks[next]; char out[256]; int ol=0;
            int ci=0; while(ci<(int)s.len){ int cp=utf8_next(s.ptr,s.len,&ci); int by=(cp<512)?CP2BYTE[cp]:-1; if(by>=0)out[ol++]=by; }
            fwrite(out,1,ol,stdout); fflush(stdout);
            if((uint64_t)next==eos) break;
        }
        tok=next;
    }
    double dt=wall()-t0;
    printf("\n");
    if(m.is_moe){ ExpertStats es; expert_store_stats(m.es,&es);
        fprintf(stderr,"[vireo-gguf] experts: %llu/%d resident (%.0f MB%s) | lookups=%llu hits=%.0f%% read=%.0fMB\n",
            (unsigned long long)es.experts_resident, m.n_experts*m.n_layers, es.resident_bytes/1048576.0,
            es.capacity_bytes?"":" uncapped",(unsigned long long)es.requests,
            es.requests?100.0*es.hits/es.requests:0, es.bytes_read/1048576.0); }
    KVStats st; kv_stats(m.kv,&st);
    fprintf(stderr,"[vireo-gguf] %d steps in %.1fs (%.1f tok/s) | KV hits=%.0f%% read=%.0fMB spilled=%.0fMB resident=%.1fMB\n",
            steps,dt,steps/dt, st.requests?100.0*st.hits/st.requests:0, st.bytes_read/1048576.0, st.bytes_spilled/1048576.0, st.resident_bytes/1048576.0);
    return 0;
}
