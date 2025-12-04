import re
import torch

SYSTEM_PROMPT = """
You are a coding engine. 
Perform calculations, expression simplifications/evaluations, equation solving, option matching (equality checks) etc. in a Wolfram Language function, 
assuming Wolfram Engine is available to execute the code and return the answer. For geoemtric problems, coordinates can be assumed as needed to construct given entities.
Enclose code within <wolfram>f := Module[{}, ...];</wolfram>.
Example 1: To find the number of circles in given image, the corresponding Wolfram code is
<wolfram>
f := Module[{},
    list1 = {Circle[{0,0}, 1], Circle[{2,2}, 1]};
    Length[list1]
];  
</wolfram>
    
Example 2: If a triangle ABC is given in an image with D and E bisecting AB and AC respectively, with side BC = 10cm, and the question is to find the length of DE, the corresponding Wolfram code is
<wolfram>
f := Module[
  {A, B, C, D, E, DElength},
  B = {0, 0};
  C = {10, 0};
  A = {5, 5};
  D = (A + B)/2;
  E = (A + C)/2;
  DElength = EuclideanDistance[D, E];
  DElength
];
</wolfram>

Example 3: If the question is to find the fraction of squares in an image containing 3 squares and 2 circles, with 4 options to select from (A: 3/5, B: 2/3, C: 2/5, D: 1/5), the corresponding Wolfram code is
<wolfram>
f := Module[{},
    shapes = {"Square","Square","Square","Circle","Circle"};
    total = Length[shapes];
    squares = Count[shapes, "Square"];
    fraction = squares/total;
    options = <|
        "A" -> 3/5,
        "B" -> 2/3,
        "C" -> 2/5,
        "D" -> 1/5
    |>;
    SelectFirst[
        Keys[options],
        fraction == options[#] &,
        None
    ]
];
</wolfram>
"""

FORMAT_PATTERN = re.compile(r"<wolfram>(.*?)</wolfram>", re.DOTALL | re.MULTILINE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_prompt(tokenizer, questions):
    """
    Convert question, images to chat template
    """
    if isinstance(questions, tuple):
        questions = [questions]

    messages_batch = []
    for question, images in questions:
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": question}
            ]},
        ]

        if images is not None:
            for i, img in enumerate(images):
                messages[1]["content"].insert(i, {"type": "image", "image": img})
        
        messages_batch.append(messages)

    inputs = tokenizer.apply_chat_template(messages_batch, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True, padding=True)
    return inputs.to(device)


def format_reward(responses):
    """
    Reward if think and answer format matches
    """
    scores = [1.0 if FORMAT_PATTERN.search(resp.strip()) else 0.0 for resp in responses]
    return torch.tensor(scores, dtype=torch.float32, device=device)
