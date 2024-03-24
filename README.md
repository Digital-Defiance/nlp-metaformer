# NLP MetaFormer - An ablation study on the transformer network 


---


## Intro

### From scaled dot product attention to metric tensor attention

To motivate the introduction of a modified attention we'll look at how the scaled dot product attention from 2017 is equivalent to a general quadratic form, and argue on the basis of interpretability and regularization for the imposition that the form be a metric. I have found that the formulas in the original paper are not very riguorous and at times, open to interpretation, so I'll be using ricci notation to fill in the gaps based on my knowledge of the code implementations from the original authors.

The transformations $Q_d^{nk}$, $K_d^{nk}$ and $V_d^{nk}$ act on the input embeddings to produce the well known keys, queries and values,

$$
q^{bnck} = Q_d^{nk} x^{bcd}
$$

$$
k^{bnck} = K_d^{nk} x^{bcd}
$$

$$
v^{bnck} = V_d^{nk} x^{bcd}
$$

The queries and keys are multiplied toguether and scaled before being softmaxed, producing the scores matrix,

$$
s^{bncc'} = \textrm{softmax}^{c'} \left ( \frac{1}{\sqrt{N_k}} q^{bnck} k^{bnc'k'} \delta_{kk'} \right ) 
$$

the use of the $N_k$ is what gives this core machanism its name, scaled dot product attention. The scores matrix is then applied to the values 


$$
t^{bnck} = s^{bncc'} v^{bnc''k} \delta_{c'c''}
$$

and the result is reflatened and projected to the original embedding space

$$
\bar t^{bcl} = t^{bnck}
$$


$$
y^{bcd} = E_l^d \bar t^{bcl}
$$

Our focus is on the step right before the application of a softmax 

$$
r^{bncc'} =  q^{bnck} k^{bnc'k'} \delta_{kk'}
$$

By substituting the operations that produced the queries and keys, we can see how the quadratic form emerges

$$
r^{bncc'} = Q_d^{nk}  K_{d'}^{nk'} \delta_{kk'} x^{bcd}   x^{bc'd'} 
$$

Defining $U^n_{dd'}=Q_d^{nk}  K_{d'}^{nk'} \delta_{kk'} $, 

$$
r^{bncc'} = U^n_{dd'} x^{bcd}   x^{bc'd'} 
$$

Disregarding training dynamics and efficiency considerations, we see that this is a complete mathematical equivalence. However, there is good reason not to keep this form. Even considering the case of using queries and keys, we see that the quadratic form is making use of $nd^2$ parameters while the original formulation uses $2ndk$, thus as long as $k < d/2$, that approach is more memory efficient.

However, it is not the most efficient reformulation that can be squeezed out of the quadratic form,


## CUDA Kernel of the Metric Tensor Attention

### Forwards Pass

Let $P^{nk}_d$ be $N_n$ learnable projections from $\mathbf R^{N_d}$ to $\mathbf R^{N_k}$ and $x^{bcd}$ a batch of $N_b$ sequences containing $N_c$ embeddings from $\mathbf R^{N_d}$. The action of these projections is expressed in Ricci notation by

$$p^{bnck} = P^{nk}_d  x^{bcd}$$

At the heart of the proposed attention mechanism is a learnable dot product of each projected embedding with each other embedding. This is achieved using $N_n$ learnable metric tensors $M^{n}_{kk'}$ and is given by

$$q^{bncc'} = M^{n}_{kk'} p^{bnck} p^{bnc'k'}$$

The metric tensor is symmetric, so we can reduce the number of computations by grouping the terms strategically,

$$q^{bncc'} = \delta^{kk'} M^n_{kk'} p^{bnck} p^{bnc'k'} + 2 \delta^{k>k'} M^n_{kk'} p^{bnck} p^{bnc'k'}$$

Let $F_N(v, w)$ be a pairing function that indexes the elements above and including the diagonal of a matrix from $\mathbf R^{N\times N}$, and $f$ and $g$ integer valued functions that retrieve the first and second argument of $F_N$, that is

$$  v = f(F_{N}(v, w)) $$

and

$$ w = g(F_{N}(v, w)) $$

Such an arrangement is easily achieved by storing two arrays to be used as a lookup table for $f$ and $g$. 

Finally, let $l=F_{N_l}(k, k')$, and define

$$ \bar M^n_{l} =  M^n_{f(l)g(l)} $$

which we use to rewrite our original expression as

$$q^{bncc'} = \delta^{f(l)g(l)} \bar M^n_{l} p^{bncf(l)} p^{bnc'f(l)} + 2 \tilde \delta^{f(l)g(l)}   \bar M^n_l p^{bncf(l)} p^{bnc'g(l)}$$

where $\tilde \delta^{f(l)g(l)} = 1 - \delta^{f(l)g(l)} $. At this point, our expression already fits quite well within a cuda kernel. Note how the $\delta$'s neatly define which expression needs to be calculated for a given value of $l$ and how easily that can be determined with an if-statement on $l$. Note that a further computational saving is unlocked with the usage of a metric tensor, since dot products are comutative it follows that $q^{bncc'} =q^{bnc'c}$, so the flattening procedure we just did for $kk'$ can be done for $cc'$. Let $u=F_{N_c}(c, c')$  and agree on the convention that when $f$ and $g$ act on $l$, they'll recover $k$ and $k'$, but when they act on $u$, they'll recover $c$ and $c'$, so we rewrite the forwards kernel as

$$\bar q^{bnu} = \delta^{f(l)g(l)} \bar M^n_{l} p^{bnf(u)f(l)} p^{bng(u)f(l)} + 2 \tilde \delta^{f(l)g(l)}   \bar M^n_l p^{bnf(u)f(l)} p^{bng(u)g(l)}$$

To take full advantage of the symmetry, we can proceed with the rest of the attention mechanism, which consists in the standard application of a softmax


$$ s^{bnu} = \textrm{softmax}^{g(u)} \left ( \frac{1}{\sqrt{N_k}} \delta^{f(l)g(l)} \bar M^n_{l} p^{bnf(u)f(l)} p^{bng(u)f(l)} + 2 \tilde \delta^{f(l)g(l)}   \bar M^n_l p^{bnf(u)f(l)} p^{bng(u)g(l)} \right ) $$

but followed by the application of the scores on the same projection 


$$ t^{bnck} = s^{bncc'} p^{bnc''k} \delta_{c'c''}  = s^{bnc}_ {c'} p^{bnc'k} $$
 
The result is then reflattened and a final transformation is applied to ensure mixing of the features and align the dimensionality to the original embedding space


$$
\bar t^{bcl} = t^{bnck}
$$


$$
y^{bcd} = E_l^d \bar t^{bcl}
$$


However, for now, we'll choose this as our forwards pass 

$$q^{bncc'} = \delta^{f(l)g(l)} \bar M^n_{l} p^{bncf(l)} p^{bnc'f(l)} + 2 \tilde \delta^{f(l)g(l)}   \bar M^n_l p^{bncf(l)} p^{bnc'g(l)}$$


### Backwards Pass

Gradient with respect with the metric coordinates:

$$\partial_{M^{n'}_ {k'''k''''}} q^{bncc'} =  \partial_{M^{n'}_{k'''k''''}}  M^{n} _{kk'}  p^{bnck} p^{bnc'k'}$$


$$\partial_{M^{n'} _ {k'''k''''}} q^{bncc'} =   p^{bnck} p^{bnc'k'} \partial_{M^{n'}_ {k'''k''''}}  M^{n}_{kk'}$$

$$
\partial_{M^{n'} _ {k'''k''''}} q^{bncc'} =   p^{bnck} p^{bnc'k'} \delta^{nn'} \delta ^ {kk'''} \delta^{k'k''''}
$$

$$
\partial_{M^{n}_{k'''k''''}} q^{bncc'} =   p^{bnck'''} p^{bnc'k''''}
$$

$$
\partial_{M^{n}_{kk'}} q^{bncc'} =   p^{bnck} p^{bnc'k'}
$$


$$
\partial_{M^n_l} q^{bnu} =   p^{bnf(u)f(l)} p^{bng(u)g(l)}
$$



Gradient with respect to the input coordinates

$$
\partial_{p^{bnc''k''}} q^{bncc'} = M^{n}_ {kk'} \partial_{p^{bnc''k''}} p^{bnck} p^{bnc'k'}
$$





$$
\partial_{p^{bnc''k''}} q^{bncc'} = M^{n} _ {kk'} \left ( p^{bnc'k'} \partial_{p^{bnc''k''}} p^{bnck}  +  p^{bnck} \partial_{p^{bnc''k''}} p^{bnc'k'} \right )
$$

$$
\partial_{p^{bnc''k''}} q^{bncc'} = M^{n}_ {kk'}  \left ( p^{bnc'k'} \delta^{c''c} \delta^{k''k}  +  p^{bnck} \delta^{c''c'} \delta^{k''k'}   \right )
$$


$$
\partial_{p^{bnc''k''}} q^{bncc'} =  M^{n} _ {kk'} p^{bnc'k'} \delta^{c''c}  \delta^{k''k}  +  M^{n}_ {kk'} p^{bnck} \delta^{c''c'} \delta^{k''k'}
$$


$$
\partial_{p^{bnc''k''}} q^{bncc'} =  M^{n} _ {k''k'} p^{bnc'k'} \delta^{c''c}    +  M^{n}_ {kk''} p^{bnck} \delta^{c''c'}
$$


$$
\partial_{p^{bnc''k''}} q^{bncc'} =  M^{n} _ {k''k'} p^{bnc'k'}  \delta^{c''c}   +  M^{n}_ {k''k} p^{bnck} \delta^{c''c'}
$$



$$
\partial_{p^{bnc''k''}} q^{bncc'} =  M^{n} _ {k''k} p^{bnc'k}  \delta^{c''c}   +  M^{n}_ {k''k} p^{bnck} \delta^{c''c'}
$$



$$
\partial_{p^{bnc''k'}} q^{bncc'} =  M^{n} _ {k'k} p^{bnc'k}  \delta^{c''c}   +  M^{n}_ {k'k} p^{bnck} \delta^{c''c'}
$$



----




## Experiments

Note: all workflows have been removed, pipelines are being moved to prefect

| Name and Status | Dataset | Usability | Workflow Badge |
|-----------------|---------|-----------|----------------|
| Sentiment Analysis Task (Completed with success) | [asa-v0.2.0](https://github.com/Digital-Defiance/llm-voice-chat/releases/tag/asa-v0.2.0) | Outdated | [![train-model: Sentiment Analysis Amazon Reviews @ EC2 Spot](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/train-model-sentiment-analysis-task-asa.yml/badge.svg?branch=main)](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/train-model-sentiment-analysis-task-asa.yml) |
| Sentiment Analysis Task (Completed without success, model overfits easily) | stanford dataset | Outdated | [![train-model: Sentiment Analysis @ EC2 Spot](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/train-model-sentiment-analysis-task.yml/badge.svg)](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/train-model-sentiment-analysis-task.yml) |
| GPT Shakespeare Textgen (Completed with success) | [sha-v0.1.0](https://github.com/Digital-Defiance/llm-voice-chat/releases/tag/sha-v0.1.0) | Outdated | [![GPT Array Sorter Experiment](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/gpt_shakespear_experiment.yml/badge.svg)](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/gpt_shakespear_experiment.yml) |
| GPT Array Sorter Experiment (Completed with success) | Generated | Outdated | [![GPT Array Sorter Experiment](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/python-app.yml/badge.svg)](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/python-app.yml) |




## Roadmap
https://github.com/orgs/Digital-Defiance/projects/11/views/1

### Phase 1

In this phase I am exploring a transformer variant and laying the groundwork for an ablation study similar to the one made on the MetaFormer (for vision). This phase is also primarily for me to build up my knowledge of NLP, data engineering, MLOps and large scale model training while hopefully getting some useful research done. 

- [x] implement and train a simple gpt that sorts tokens - [![GPT Array Sorter Experiment](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/python-app.yml/badge.svg)](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/python-app.yml)
- [x] use simpler implementation to contruct the MLOps infra
- [x] train a larger gpt on shakespeare - [![GPT Array Sorter Experiment](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/gpt_shakespear_experiment.yml/badge.svg)](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/gpt_shakespear_experiment.yml)
- [x] experiment with transformer modifications (i.e. mtn)
- [ ] perform systematic comparison between mtn and transformer
  - [ ] Sentiment Analysis 
    - [x] [![train-model: Sentiment Analysis @ EC2 Spot](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/train-model-sentiment-analysis-task.yml/badge.svg)](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/train-model-sentiment-analysis-task.yml)
    - [x]  [![train-model: Sentiment Analysis Amazon Reviews @ EC2 Spot](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/train-model-sentiment-analysis-task-asa.yml/badge.svg?branch=main)](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/train-model-sentiment-analysis-task-asa.yml)
  - [ ] topic classification
  - [ ] machine translation
  - [ ] summarization 
- [ ] write report on comparison between transformer and metric tensor network (might focus more on this depending on the results)


### Phase 2 

In this phase the plan is to deploy the models trained on phase 1, alongside with open source LLMs.

- [ ] write webapp (traefik - go + htmx + tmpl - fastapi + models)
- [ ] deploy webapp
- [ ] release first version

### Phase 3

In this phase, all the lessons from 1 and 2 will be used to to fine tune Lamma into multi-modility and finally, non-turn based voice chat. 

- [ ] fine tune gpt2
- [ ] fine tune lamma
- [ ] setup whisper




## possible dependencies

- https://github.com/mozilla/TTS

## some literature

- https://paperswithcode.com/method/strided-attention
- https://paperswithcode.com/method/fixed-factorized-attention
- https://paperswithcode.com/method/dot-product-attention
- https://paperswithcode.com/method/scaled

## datasets

- https://paperswithcode.com/dataset/cnn-daily-mail-1
- https://metatext.io/datasets/wikisummary

## The reasoning behind modifying transformers self attention 

NOTE: WIP

NOTE2: this is not the usual index notation, see next section for explanation

In the proposed self-attention mechanism, we consider a sequence input represented by a tensor $x_{bwc}$, where $b$ indexes the batch size, $w$ the sequence length, and $c$ the feature dimensions. The mechanism leverages a metric tensor to enhance the geometric understanding of the attention process proposed in 2017.

The first step involves a series of linear transformation of $x_{bwc}$ to $n$ lower-dimensional spaces. For each head $n$, this is achieved using a weight tensor $A_{ck}^{(n)}$ where $k = c / n$ represents the reduced dimensions for each head. The transformation is given by:

$$z_{bwk}^{(n)} = x_{bwc} A_{ck}^{(n)} $$


The heart of the mechanism lies in the metric tensor $G^{(n)} _ {kk}$, initialized as a product of a learnable, lower triangular tensor $P ^{(n)} _ {kk}$ and its transpose. This ensures that $G^{(n)} _ {kk}$ is symmetric and semi-positive definite:
$$G^{(n)}_{kk} = P ^{(n)} _ {kk} (P ^{(n)} _ {kk})^T$$
This introduces a geometric structure into the attention mechanism. The tensor $G^{(n)} _ {kk}$ allowes the network to construct custom dot products which can be calculated via the usual quadratic form,

$$ \textrm{dot}^{(n)}(z^{(n)} _ {bwk}, z^{(n)} _ {bwk}) = z^{(n)} _ {bwk} G^{(n)} _ {kk} ( z^{(n)} _ {bwk} ) ^T$$

We use this custom metric to replace the $W_qW_k^T$ shown in the original 2017 publication,

$$
S^{(n)}_ {bww} =
\text{softmax}_k\left( \frac{
\textrm{dot}^{(n)}(z^{(n)} _ {bwk}, z^{(n)} _ {bwk})
}{\sqrt{K}} \right)
$$

Here, $S^{(n)} _ {bww}$ represents the attention scores, quantifying the influence of each word in the sequence on every other word, with $w'$ indexing the sequence length. Once the attention scores are obtained, they are used to compute the output for each head. The output for head $n$, $O^{(n)}_{bwk}$, is a weighted sum of the transformed features:

$$O^{(n)}_ {bwk} = S^{(n)} _{bww} z^{(n)} _{bwk}$$

Finally, the outputs from all heads are concatenated and passed through another linear transformation $B_{ij}$ to yield the final output $Y_{bwi}$:

$$Y _{bwi} = B _{ij} \left[ O^{(1)} _ {bwj}, O^{(2)} _{bwj}, \ldots, O^{(N)} _{bwj} \right]$$

This mechanism, through the use of the metric tensor $G^{(n)}_{kk}$, provides a novel approach to compute attention, offering a geometric perspective to the understanding and processing of sequences in neural networks.



## Tensor Notation Guidelines

In our code, we use a specific notation to denote the shape of tensors. Here's how it works:

- A tensor's shape is indicated by appending a suffix to the variable name. Each letter in the suffix corresponds to a dimension in the tensor's shape. For example, a tensor with shape `(a, b, c)` would be named `some_tensor_abc`:

    ```python
    a, b, c = 10, 3, 5
    some_tensor_abc = torch.randn(a, b, c)
    ```

- If the dimensions of the tensor are not represented by single letters, we use their initials. For instance, a tensor with dimensions `batch_size` and `vocabolary_size` would be named `some_tensor_bv`:

    ```python
    batch_size, vocabolary_size = 32, 1024
    some_tensor_bv = torch.randn(batch_size, vocabolary_size)
    ```

- If a dimension has an explicit value, we include that value in the suffix. For example, `some_tensor_b2ac` indicates that the tensor has a second dimension (`dim=1`) with a size of 2. We only include explicit values in the suffix if they have more than one digit.

- We also extend this notation to functions. A function name like `some_function_tq` indicates that the function transforms dimension `q` into size `t`:

    ```python
    result_abct = some_function_tq(input_abcq)
    ```

This notation helps us keep track of tensor shapes throughout our code, making it easier to understand and debug.



# Preliminary Results

## Modified Self Attention, Metric Tensor Heads

![image](https://github.com/Digital-Defiance/llm-voice-chat/assets/63464503/5f17ae14-a627-4c0d-9a44-6b60e69f3774)

![image](https://github.com/Digital-Defiance/llm-voice-chat/assets/63464503/3bec2b7d-a47b-48bf-a7e0-b8a7c293a9e9)

![image](https://github.com/Digital-Defiance/llm-voice-chat/assets/63464503/b8026426-9d97-4379-8e08-f6c5a4722206)

## Loss Graph Comparison between Transformer and Metric Tensor Network

![2024-01-03-052123_571x464_scrot](https://github.com/Digital-Defiance/llm-voice-chat/assets/63464503/94534309-d07b-4ad2-9a87-9dcd23f012a2)

## Output Comparison

### Transformer:

```
The meaning of life is full of me:
if
I spy an age be content: the sea excuse that very
are to achieve for our prisoner to Rome's wife,
'Sirrah's see, command, let twenty pound
Strive might now; since is about me than,
Were then but the point of death: he were a
them where I'll wear what to wash you, for
And copy of the action; and gave me down himself
For why I should give for these fashion of them
Whether but relished to the hand:
Then speak, old and pray, no when the petition
With what, by our petition you bear this, after;
Not writ we held him. When like subjects put out,
That six years would reap the will we more
To follow us fly the throne of heaven, as sad
Which had no further. There, gentle Paulina,
The same no bodes with our valiant power,
And that's the herd, there be an his certain
Nor private gentlemen with you,--O, and your
Long may answer, from us, to fly seeing remorse,
The mutinous their eyes, who hath slain!
His senate-face, and my life sent,
The dangerous lenity where she starts;
And all with the sea or mistaken;
For him from whence can I do.

SOMERSET:
No310 of fear, here comes it.

ARCHBUSHY:
Ay, give? it not fall of this:
If thy mother shall be seen the world
Might gently before thyself in time.
MeDecline image and look'd, then, take him:
'Shall I we see thee thy tongue.

GREEN:
All Edward again. Give me to France, madam, I.
```


### metric tensor net

```
The meaning of life is soaking,'er's friend,
For I will in some man. It were to Richmond,
But by the general made up,
And when he walks, make him yea,
Thou shalt teach thee will to give himself?
Than Lewis he did I think of infirm'd too.

HASTINGS:
Under whom me so I swear to deliver me?

HASTINGS:

Ghost that I, a kingdom this amongst us.

BUCKINGHAM:
His lie such an Cates, he fears you.

KING EDWARD IV:
But raise this stands giftedave.

QUEEN MARGARET:
The rest be not your crown?

QUEEN ELIZABETH:
Is this once, that I enforce his sign of four
Which be uncle, till I let me to have done,
And not privy friend to a grief weep.
An, and my husband's wife hath done a want of mine.
My frost may follow to love.

Y ANNE:
The high forehead Margaret of Warwick mans your tongue and Derby,
To prove it of Buckingham shall way the streets.

QUEEN ELIZABETH:
Ay, by this device are butcher of Glouces;
Poor high love kill it will--

QUEEN ELIZABETH: may awake Boling;
And unblown, unto the cause
Or once to her repeal'd in private.
InsTER:
Come, no, the dying sovereign to my son and this land what
And were for Edward to thither to kill'd.
The knights and no conquest of them?
But do you be nor bestow' sovereign, nor debt:
Our children of Clarence, if 'tis trueborn blood.
Thus till then, my Edward is like our course of scful!
```

## preliminary conclusion

these results do not eliminate the modified network as a possible alternative to the transformer, but no significant advantages were found during training

according to gpt4, the output from the metric tensor net looks more coherent (I don't fully understand archaic english) but this is likely to be a coincidence

the metric tensor network was obtained from an ad hoc modification to the transformer, it does not yet make use of the reduced number of parameters to increase efficiency

----> the same results were obtained for the new architecture while making use of less parameters, could be a promising direction, more testing is needed. interpretability was not fully evaluated





