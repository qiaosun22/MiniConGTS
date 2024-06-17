from data.data_preparing import Instance

def load_data_instances(tokenizer, sentence_packs, args):
    instances = []
    for sentence_pack in sentence_packs:
        instances.append(Instance(tokenizer, sentence_pack, args))
    return instances
