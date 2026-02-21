def count_parameters(model, is_print=False):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_print:
        print(f"Trainable parameters: {total_params / 1e6:.2f}M ({total_params / 1e9:.2f}B)")
    return total_params