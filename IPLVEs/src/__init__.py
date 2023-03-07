def load_model(args):
    if args.model.model_alias == 'bert':
        from .bert_word_emb import load_model
    elif args.model.model_alias == 'gpt2':
        from .gpt2_word_emb import load_model
    elif args.model.model_alias == 'opt':
        from .opt_word_emb import load_model
    else:
        raise NotImplementedError
    return load_model(args)


def average_word_embedding(contexts, model, tokenizer, args):
    if args.model.model_alias == 'bert':
        from .bert_word_emb import average_word_embedding
    elif args.model.model_alias == 'gpt2':
        from .gpt2_word_emb import average_word_embedding
    elif args.model.model_alias == 'opt':
        from .gpt2_word_emb import average_word_embedding
    else:
        raise NotImplementedError
    return average_word_embedding(contexts, model, tokenizer, args)