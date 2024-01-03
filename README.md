# llm-voice-chat

[![GPT Array Sorter Experiment](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/python-app.yml/badge.svg)](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/python-app.yml)

[![GPT Array Sorter Experiment](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/gpt_shakespear_experiment.yml/badge.svg)](https://github.com/Digital-Defiance/llm-voice-chat/actions/workflows/gpt_shakespear_experiment.yml)


Speak with a language model.

## Roadmap

phase 1 - exploratory

- [x] implement and train a simple gpt that sorts tokens
- [x] use simpler implementation to contruct the MLOps infra
- [x] train a larger gpt on shakespeare
- [ ] experiment with transformer modifications (i.e. mtn)
- [ ] write report on comparison between transformer and metric tensor network (might focus more on this depending on the results)
- [ ] fine tune gpt2
- [ ] setup whisper

phase 2 - development

- [ ] write webapp (traefik - go + htmx + tmpl - fastapi + models)
- [ ] deploy webapp
- [ ] release first version

phase 3 - exploratory 2

- TBD




## The setup


![2024-01-01-035433_406x944_scrot](https://github.com/Digital-Defiance/llm-voice-chat/assets/63464503/bbb1fee1-78b2-4d0f-8c8b-f06e5a1a1b70)

![2024-01-01-040953_1478x706_scrot](https://github.com/Digital-Defiance/llm-voice-chat/assets/63464503/19205fbb-5d43-45b3-9947-f8403e0ebfd6)

![2024-01-01-045640_1483x757_scrot](https://github.com/Digital-Defiance/llm-voice-chat/assets/63464503/d6fb75af-7e2e-40fc-9758-e1c93981cb47)

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





