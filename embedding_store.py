from dpu_utils.mlutils import Vocabulary
from torch import nn
from configs import START, END,  MAX_VOCAB_SIZE
from utils.diff_utils import get_edit_keywords


class EmbeddingStore(nn.Module):
    def __init__(self, nl_threshold, nl_embedding_size, nl_token_counter,
                 code_threshold, code_embedding_size, code_token_counter,
                 ast_threshold, ast_embedding_size, ast_token_counter,
                 dropout_rate, load_pretrained_embeddings=False):

        super(EmbeddingStore, self).__init__()
        edit_keywords = get_edit_keywords()
        self.__nl_vocabulary = Vocabulary.create_vocabulary(tokens=edit_keywords,
                                                         max_size=MAX_VOCAB_SIZE,
                                                         count_threshold=1,
                                                         add_pad=True)
        self.__nl_vocabulary.update(nl_token_counter, MAX_VOCAB_SIZE, nl_threshold)
        self.__nl_embedding_layer = nn.Embedding(num_embeddings=len(self.__nl_vocabulary),
                                        embedding_dim=nl_embedding_size,
                                        padding_idx=self.__nl_vocabulary.get_id_or_unk(Vocabulary.get_pad()))
        self.nl_embedding_dropout_layer = nn.Dropout(p=dropout_rate)

        self.__code_vocabulary = Vocabulary.create_vocabulary(tokens=edit_keywords,
                                                    max_size=MAX_VOCAB_SIZE,
                                                    count_threshold=1,
                                                    add_pad=True)
        self.__code_vocabulary.update(code_token_counter, MAX_VOCAB_SIZE, code_threshold)
        self.__code_embedding_layer = nn.Embedding(num_embeddings=len(self.__code_vocabulary),
                        embedding_dim=code_embedding_size,
                        padding_idx=self.__code_vocabulary.get_id_or_unk(
                        Vocabulary.get_pad()))
        self.code_embedding_dropout_layer = nn.Dropout(p=dropout_rate)

        self.__ast_vocabulary = Vocabulary.create_vocabulary(tokens=['add', 'delete', 'move', 'update'],
                                                         max_size=MAX_VOCAB_SIZE,
                                                         count_threshold=1,
                                                         add_pad=True)
        self.__ast_vocabulary.update(ast_token_counter, MAX_VOCAB_SIZE, ast_threshold)
        self.__ast_embedding_layer = nn.Embedding(num_embeddings=len(self.__ast_vocabulary),
                                                   embedding_dim=ast_embedding_size,
                                                   padding_idx=self.__ast_vocabulary.get_id_or_unk(
                                                       Vocabulary.get_pad()))
        self.ast_embedding_dropout_layer = nn.Dropout(p=dropout_rate)

        print('NL vocabulary size: {}'.format(len(self.__nl_vocabulary)))
        print('Code vocabulary size: {}'.format(len(self.__code_vocabulary)))
        print('AST vocabulary size: {}'.format(len(self.__ast_vocabulary)))


    def get_nl_embeddings(self, token_ids):
        return self.nl_embedding_dropout_layer(self.__nl_embedding_layer(token_ids))
    
    def get_code_embeddings(self, token_ids):
        return self.code_embedding_dropout_layer(self.__code_embedding_layer(token_ids))

    def get_ast_embeddings(self, token_ids):
        return self.ast_embedding_dropout_layer(self.__ast_embedding_layer(token_ids))
    
    @property
    def nl_vocabulary(self):
        return self.__nl_vocabulary
    
    @property
    def code_vocabulary(self):
        return self.__code_vocabulary

    @property
    def ast_vocabulary(self):
        return self.__ast_vocabulary

    @property
    def nl_embedding_layer(self):
        return self.__nl_embedding_layer

    @property
    def code_embedding_layer(self):
        return self.__code_embedding_layer

    @property
    def ast_embedding_layer(self):
        return self.__ast_embedding_layer

    # span_diff_code_tokens
    def get_padded_code_ids(self, code_sequence, pad_length):
        return self.__code_vocabulary.get_id_or_unk_multiple(code_sequence,
                                    pad_to_size=pad_length,
                                    padding_element=self.__code_vocabulary.get_id_or_unk(Vocabulary.get_pad()))
    
    def get_padded_nl_ids(self, nl_sequence, pad_length):
        return self.__nl_vocabulary.get_id_or_unk_multiple(nl_sequence,
                                    pad_to_size=pad_length,
                                    padding_element=self.__nl_vocabulary.get_id_or_unk(Vocabulary.get_pad()))

    def get_padded_ast_ids(self, ast_sequence, pad_length):
        return self.__ast_vocabulary.get_id_or_unk_multiple(ast_sequence,
                                    pad_to_size=pad_length,
                                    padding_element=self.__ast_vocabulary.get_id_or_unk(Vocabulary.get_pad()))
    
    def get_extended_padded_nl_ids(self, nl_sequence, pad_length, inp_ids, inp_tokens):
        nl_ids = []
        for token in nl_sequence:
            nl_id = self.get_nl_id(token)
            if self.is_nl_unk(nl_id) and token in inp_tokens:
                copy_idx = inp_tokens.index(token)
                nl_id = inp_ids[copy_idx]
            nl_ids.append(nl_id)
        
        if len(nl_ids) > pad_length:
            return nl_ids[:pad_length]
        else:
            padding = [self.__nl_vocabulary.get_id_or_unk(Vocabulary.get_pad())] * (pad_length - len(nl_ids))
            return nl_ids + padding
    
    def get_code_id(self, token):
        return self.__code_vocabulary.get_id_or_unk(token)
    
    def is_code_unk(self, id):
        return id == self.__code_vocabulary.get_id_or_unk(Vocabulary.get_unk())
    
    def get_code_token(self, token_id):
        return self.__code_vocabulary.get_name_for_id(token_id)
    
    def get_nl_id(self, token):
        return self.__nl_vocabulary.get_id_or_unk(token)
    
    def is_nl_unk(self, id):
        return id == self.__nl_vocabulary.get_id_or_unk(Vocabulary.get_unk())
    
    def get_nl_token(self, token_id):
        return self.__nl_vocabulary.get_name_for_id(token_id)
    
    def get_vocab_extended_nl_token(self, token_id, inp_ids, inp_tokens):
        if token_id < len(self.__nl_vocabulary):
            return self.get_nl_token(token_id)
        elif token_id in inp_ids:
            copy_idx = inp_ids.index(token_id)
            return inp_tokens[copy_idx]
        else:
            return Vocabulary.get_unk()
    
    def get_nl_tokens(self, token_ids, inp_ids, inp_tokens):
        tokens = [self.get_vocab_extended_nl_token(t, inp_ids, inp_tokens) for t in token_ids]
        if END in tokens:
            return tokens[:tokens.index(END)]
        return tokens
    
    def get_end_id(self):
        return self.get_nl_id(END)
