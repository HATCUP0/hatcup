START = '<sos>'
END = '<eos>'
NL_EMBEDDING_SIZE = 64
CODE_EMBEDDING_SIZE = 64
AST_EMBEDDING_SIZE = 64
HIDDEN_SIZE = 64
DECODER_HIDDEN_SIZE = 128
DROPOUT_RATE = 0.6
NUM_LAYERS = 2
LR = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10
VOCAB_CUTOFF_PCT = 5
LENGTH_CUTOFF_PCT = 95
NUM_ENCODERS = 3
MAX_VOCAB_EXTENSION = 50
BEAM_SIZE = 10
MAX_RETURN_TYPE_LENGTH = 10
MAX_VOCAB_SIZE = 25000
PREDICTION_DIR = 'prediction'
MODEL_LAMBDA = 0.5
LIKELIHOOD_LAMBDA = 0.3
OLD_METEOR_LAMBDA = 0.2
GEN_MODEL_LAMBDA = 0.5
GEN_OLD_BLEU_LAMBDA = 0.5
