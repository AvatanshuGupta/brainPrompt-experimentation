from transformers import LlamaTokenizer, LlamaModel, AutoTokenizer, AutoModelForCausalLM
import torch
from llm2vec.models import LlamaBiModel
import tqdm
import clip


# Use https://huggingface.co/knowledgator/Llama-encoder-1.0B
def llama_encoder(model, tokenizer, text, datatoken=''):

    inputs = tokenizer(text, return_tensors="pt")

    if datatoken == 'label':
        begin_idx = 6
        end_idx = (inputs['input_ids'].squeeze() == 4967).nonzero(as_tuple=True)[0].item()
        # print(end_idx)
    elif datatoken == 'ROI':
        begin_idx = 1
        end_idx = (inputs['input_ids'].squeeze() == 29901).nonzero(as_tuple=True)[0].item()
    else:
        begin_idx = 0
        end_idx = -1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state.squeeze()
    emb = last_hidden_states[begin_idx: end_idx].mean(dim=0)
    return emb


def clip_encoder(model, text, device):

    text = clip.tokenize(text).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features[0]


def generate_embeddings(path, out_path=None, model='llama', datatoken=''):
    if model == 'llama':
        tokenizer = AutoTokenizer.from_pretrained("knowledgator/Llama-encoder-1.0B")
        lm = LlamaBiModel.from_pretrained("knowledgator/Llama-encoder-1.0B")
    elif model == 'clip':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lm, _ = clip.load("ViT-B/32", device=device)
    else:
        raise ValueError("Invalid model name.")

    # Determine output path
    if out_path is None:
        out_path = path[:-4] + (".pt" if model == 'llama' else "_clip.pt")

    f = open(path, 'r')
    embeddings = []
    for line in tqdm.tqdm(f.readlines()):
        if model == 'llama':
            embeddings.append(llama_encoder(lm, tokenizer, line, datatoken=datatoken).cpu())
        elif model == 'clip':
            embeddings.append(clip_encoder(lm, line, device).cpu())
    f.close()

    torch.save(embeddings, out_path)
    print(f"Saved {len(embeddings)} embeddings to {out_path}")
    return embeddings


# ── Generate all embeddings ──────────────────────────────────────────────────
# Label embeddings (2 embeddings, one per class)
# generate_embeddings("data/prompts/label_prompts/abide_label.txt",
#                     datatoken='label')

# ROI embeddings (116 embeddings)
# generate_embeddings("data/prompts/ROI_prompts/AAL116_short.txt",
#                     datatoken='ROI')

# Subject-level demographic embeddings — NO label info (1025 embeddings)
generate_embeddings(
    path     = "data/prompts/meta_prompts/abide_meta_descriptions.txt",
    out_path = "data/prompts/meta_prompts/abide_meta_datatoken.pt",
    datatoken = ''
)

# generate_embeddings("data/prompts/ROI_prompts/AAL116_short.txt", datatoken='ROI')

#generate_embeddings("data/prompts/label_prompts/abide_label.txt", datatoken='label')
# At the bottom of text2emb.py, comment out the current line and add:

# generate_embeddings("data/prompts/label_prompts/abide_label.txt", datatoken='label')  # original
# generate_embeddings("data/prompts/meta_prompts/abide_meta_descriptions.txt", datatoken='')