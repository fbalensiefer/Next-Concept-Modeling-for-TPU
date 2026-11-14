"""Analytic FLOP estimates, not compiler estimates or measured hardware counters.

MAC=2. Dense attention is charged at stored capacity, including masked slots.
Elementwise norm/activation/softmax terms are approximate; RNG, indexing,
communication and memory traffic are excluded. Training backward=2*forward,
Adam=10*parameter_count. Counts deliberately expose conventions rather than
calling a loop-body compiler cost an exact execution FLOP count.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class FlopModel:
    width:int
    layers:int
    heads:int
    mlp_ratio:int=4
    vocabulary:int=256


def transformer_forward(config,batch,length,*,head=True):
    w,r,h=config.width,config.mlp_ratio,config.heads
    attention=(8+4*r)*batch*length*w*w+4*batch*length*length*w
    elementwise=20*batch*length*w+8*batch*length*r*w+5*batch*h*length*length
    output=(2*batch*length*w*config.vocabulary+5*batch*length*config.vocabulary) if head else 0
    return float(config.layers*(attention+elementwise)+output)


def ar_training_step(config,batch,length,parameters):
    return 3*transformer_forward(config,batch,length)+10*parameters


def ar_cached_decode_step(config,batch,capacity,*,head=True):
    w,r=config.width,config.mlp_ratio
    return float(config.layers*((8+4*r)*batch*w*w+4*batch*capacity*w+
        20*batch*w+8*batch*r*w+5*batch*config.heads*capacity)+(2*batch*w*config.vocabulary+5*batch*config.vocabulary)*int(head))


def ar_cache_prefill_step(config,batch,capacity):
    """Cache-only step: the last layer needs K/V, not its attention/MLP.

    Earlier layers still need their full output to construct the next layer's
    K/V. Charge the fused QKV projection nominally (a compiler may prune Q).
    """
    layer=ar_cached_decode_step(config,batch,capacity,head=False)/config.layers
    last_projection=6*batch*config.width**2+10*batch*config.width
    return float((config.layers-1)*layer+last_projection)


def ar_cached_generation(config,batch,capacity,prompt_len,new_tokens):
    return prompt_len*ar_cache_prefill_step(config,batch,capacity)+new_tokens*ar_cached_decode_step(config,batch,capacity)


def codec_forward(batch,length,dim,vocabulary=256,*,levels=3,codes=64,lengths=None,encoder=True,decoder=True):
    lengths=lengths or (length,)*levels
    conv_block=16*batch*length*dim*dim+28*batch*length*dim
    count=2*conv_block*int(encoder)+2*conv_block*int(decoder)
    # The RVQ scan retains L storage positions at every level, including padding.
    if encoder:count+=2*batch*levels*length*codes*dim+4*levels*batch*length*length*dim
    if decoder:count+=2*batch*length*dim*vocabulary+5*batch*length*vocabulary
    return float(count)


def ncm_codec_step(batch,length,dim,parameters,**kwargs):
    return 3*codec_forward(batch,length,dim,**kwargs)+10*parameters


def ncm_prior_forward(config,batch,lengths,dim,prompt_len,ar_coarse_levels=0,*,cached=False):
    count=0.
    for k,n in enumerate(lengths):
        if cached and k<ar_coarse_levels:
            count+=prompt_len*ar_cache_prefill_step(config,batch,prompt_len+n)+n*ar_cached_decode_step(config,batch,prompt_len+n)
        else:count+=transformer_forward(config,batch,prompt_len+n,head=False)+2*batch*n*config.width*config.vocabulary+5*batch*n*config.vocabulary
        count+=2*batch*n*dim*config.width+4*batch*n*max(lengths)*dim
    return count


def ncm_prior_step(config,batch,lengths,dim,prompt_len,parameters,*,ar_coarse_levels=0,
                   codes=64,teacher=None,teacher_length=None,decoder_trainable=False):
    count=3*ncm_prior_forward(config,batch,lengths,dim,prompt_len,ar_coarse_levels)+10*parameters
    count+=codec_forward(batch,max(lengths),dim,levels=len(lengths),codes=codes,lengths=lengths,decoder=False)
    count+=(3 if decoder_trainable else 2)*codec_forward(batch,max(lengths),dim,encoder=False)
    if teacher is not None:count+=transformer_forward(teacher,batch,teacher_length)
    return count


def speculative_round(draft,verify,rounds):
    return dict(analytic_flops_per_round=float(draft+verify),measured_rounds=float(rounds),
                analytic_flops=float((draft+verify)*rounds),analytic_label='analytic estimate')
