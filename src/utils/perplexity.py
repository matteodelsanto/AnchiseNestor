import torch


def compute_global_perplexity(text, model, tokenizer, window_size=20, device="cpu", leap=0):
    encodings = tokenizer(text, return_tensors='pt').to(device)

    max_length = window_size if window_size > 0 else model.config.n_positions
    stride = 1
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
            
    ppl = torch.exp(torch.stack(nlls[leap:]).sum() / (end_loc-leap))
    
    # Sposta il tensore su CPU prima di convertirlo in numpy
    return ppl.cpu().numpy().tolist()