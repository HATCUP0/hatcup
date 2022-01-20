import argparse
import os
from collections import Counter
import random
from torch import nn
from tqdm import tqdm
from configs import *
from utils.data_utils import read_examples_from_file, UpdateBatchData, Example
from utils import diff_utils
from utils.tensor_utils import *
from utils.predict_utils import *
from embedding_store import EmbeddingStore
from encoder import Encoder
from external_cache import get_code_features,get_nl_features,get_ast_features, NUM_CODE_FEATURES, NUM_NL_FEATURES, NUM_AST_FEATURES
from update_decoder import UpdateDecoder


class CommentUpdateModel(nn.Module):
    """Edit model which learns to map a sequence of code edits to a sequence of comment edits and then applies the edits to the
       old comment in order to produce an updated comment."""
    def __init__(self, model_path):
        super(CommentUpdateModel, self).__init__()
        self.model_path = model_path
        self.torch_device_name = 'cpu'
    
    def initialize(self, train_data):
        """Initializes model parameters from pre-defined hyperparameters and other hyperparameters
           that are computed based on statistics over the training data."""
        nl_lengths = []
        code_lengths = []
        ast_lengths = []
        nl_token_counter = Counter()
        code_token_counter = Counter()
        ast_token_counter = Counter()

        for ex in train_data:
            trg_sequence = [START] + ex.span_minimal_diff_comment_tokens + [END]
            nl_token_counter.update(trg_sequence)
            nl_lengths.append(len(trg_sequence))

            old_nl_sequence = ex.old_comment_tokens
            nl_token_counter.update(old_nl_sequence)
            nl_lengths.append(len(old_nl_sequence))

            code_sequence = ex.span_diff_code_tokens
            code_token_counter.update(code_sequence)
            code_lengths.append(len(code_sequence))

            ast_sequence = ex.ast_diff_tokens
            ast_token_counter.update(ast_sequence)
            ast_lengths.append(len(ast_sequence))

        self.max_nl_length = int(np.percentile(np.asarray(sorted(nl_lengths)), LENGTH_CUTOFF_PCT))
        self.max_code_length = int(np.percentile(np.asarray(sorted(code_lengths)), LENGTH_CUTOFF_PCT))
        self.max_ast_length = int(np.percentile(np.asarray(sorted(ast_lengths)), LENGTH_CUTOFF_PCT))
        self.max_vocab_extension = self.max_nl_length + self.max_code_length + self.max_ast_length
    
        nl_counts = np.asarray(sorted(nl_token_counter.values()))
        nl_threshold = int(np.percentile(nl_counts, VOCAB_CUTOFF_PCT)) + 1
        print("nl_threshold:", nl_threshold)
        code_counts = np.asarray(sorted(code_token_counter.values()))
        code_threshold = int(np.percentile(code_counts, VOCAB_CUTOFF_PCT)) + 1
        print("code_threshold:", code_threshold)
        ast_counts = np.asarray(sorted(ast_token_counter.values()))
        ast_threshold = int(np.percentile(ast_counts, VOCAB_CUTOFF_PCT)) + 1

        self.embedding_store = EmbeddingStore(nl_threshold, NL_EMBEDDING_SIZE, nl_token_counter,
                                              code_threshold, CODE_EMBEDDING_SIZE, code_token_counter,
                                              ast_threshold, AST_EMBEDDING_SIZE, ast_token_counter,
                                              DROPOUT_RATE, False)
        self.code_encoder = Encoder(CODE_EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE)
        self.nl_encoder = Encoder(NL_EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE)
        self.ast_encoder = Encoder(AST_EMBEDDING_SIZE, HIDDEN_SIZE, NUM_LAYERS, 0.2)
        self.decoder = UpdateDecoder(NL_EMBEDDING_SIZE, DECODER_HIDDEN_SIZE,
            2*HIDDEN_SIZE, self.embedding_store, NL_EMBEDDING_SIZE, DROPOUT_RATE)
        self.encoder_final_to_decoder_initial = nn.Parameter(torch.randn(2*NUM_ENCODERS*HIDDEN_SIZE,
            DECODER_HIDDEN_SIZE, dtype=torch.float, requires_grad=True))

        self.code_features_to_embedding = nn.Linear(CODE_EMBEDDING_SIZE + NUM_CODE_FEATURES,
                                                    CODE_EMBEDDING_SIZE, bias=False)
        self.nl_features_to_embedding = nn.Linear(NL_EMBEDDING_SIZE + NUM_NL_FEATURES,
                                                  NL_EMBEDDING_SIZE, bias=False)
        self.ast_features_to_embedding = nn.Linear(AST_EMBEDDING_SIZE + NUM_AST_FEATURES,
                                                   AST_EMBEDDING_SIZE, bias=False)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR)

    def get_batches(self, dataset, shuffle=True):
        """Divides the dataset into batches based on pre-defined BATCH_SIZE hyperparameter.
           Each batch is tensorized so that it can be directly passed into the network."""
        batches = []
        if shuffle:
            random.shuffle(dataset)

        for start_idx in tqdm(range(0, len(dataset), BATCH_SIZE)):
            end_idx = min(start_idx + BATCH_SIZE, len(dataset))
            
            code_token_ids = []
            code_lengths = []
            old_nl_token_ids = []
            old_nl_lengths = []
            ast_token_ids = []
            ast_lengths = []
            trg_token_ids = []
            trg_extended_token_ids = []
            trg_lengths = []
            invalid_copy_positions = []
            inp_str_reps = []
            inp_ids = []
            code_features = []
            nl_features = []
            ast_features = []

            for i in range(start_idx, end_idx):
                code_sequence = dataset[i].span_diff_code_tokens
                code_sequence_ids = self.embedding_store.get_padded_code_ids(code_sequence, self.max_code_length)
                code_length = min(len(code_sequence), self.max_code_length)
                code_token_ids.append(code_sequence_ids)
                code_lengths.append(code_length)

                old_nl_sequence = dataset[i].old_comment_tokens
                old_nl_length = min(len(old_nl_sequence), self.max_nl_length)
                old_nl_sequence_ids = self.embedding_store.get_padded_nl_ids(old_nl_sequence, self.max_nl_length)
                old_nl_token_ids.append(old_nl_sequence_ids)
                old_nl_lengths.append(old_nl_length)

                ast_sequence = dataset[i].ast_diff_tokens
                ast_length = min(len(ast_sequence), self.max_ast_length)
                ast_sequence_ids = self.embedding_store.get_padded_code_ids(ast_sequence, self.max_ast_length)
                ast_token_ids.append(ast_sequence_ids)
                ast_lengths.append(ast_length)
                
                ex_inp_str_reps = []
                ex_inp_ids = []
                
                extra_counter = len(self.embedding_store.nl_vocabulary)
                max_limit = len(self.embedding_store.nl_vocabulary) + self.max_vocab_extension
                out_ids = set()

                copy_inputs = code_sequence[:code_length] + old_nl_sequence[:old_nl_length] + ast_sequence[:ast_length]
                for c in copy_inputs:
                    nl_id = self.embedding_store.get_nl_id(c)
                    if self.embedding_store.is_nl_unk(nl_id) and extra_counter < max_limit:
                        if c in ex_inp_str_reps:
                            nl_id = ex_inp_ids[ex_inp_str_reps.index(c)]
                        else:
                            nl_id = extra_counter
                            extra_counter += 1

                    out_ids.add(nl_id)
                    ex_inp_str_reps.append(c)
                    ex_inp_ids.append(nl_id)
                
                trg_sequence = [START] + dataset[i].span_minimal_diff_comment_tokens + [END]
                trg_sequence_ids = self.embedding_store.get_padded_nl_ids(
                    trg_sequence, self.max_nl_length)
                trg_extended_sequence_ids = self.embedding_store.get_extended_padded_nl_ids(
                    trg_sequence, self.max_nl_length, ex_inp_ids, ex_inp_str_reps)
                
                trg_token_ids.append(trg_sequence_ids)
                trg_extended_token_ids.append(trg_extended_sequence_ids)
                trg_lengths.append(min(len(trg_sequence), self.max_nl_length))
                inp_str_reps.append(ex_inp_str_reps)
                inp_ids.append(ex_inp_ids)

                invalid_copy_positions.append(get_invalid_copy_locations(ex_inp_str_reps, self.max_vocab_extension,
                    trg_sequence, self.max_nl_length))
                code_features.append(get_code_features(code_sequence, dataset[i], self.max_code_length))
                nl_features.append(get_nl_features(old_nl_sequence, dataset[i], self.max_nl_length))
                ast_features.append(get_ast_features(ast_sequence, dataset[i], self.max_ast_length))

            batches.append(
                UpdateBatchData(torch.tensor(code_token_ids, dtype=torch.int64, device=self.get_device()),
                               torch.tensor(code_lengths, dtype=torch.int64, device=self.get_device()),
                               torch.tensor(old_nl_token_ids, dtype=torch.int64, device=self.get_device()),
                               torch.tensor(old_nl_lengths, dtype=torch.int64, device=self.get_device()),
                               torch.tensor(ast_token_ids, dtype=torch.int64, device=self.get_device()),
                               torch.tensor(ast_lengths, dtype=torch.int64, device=self.get_device()),
                               torch.tensor(trg_token_ids, dtype=torch.int64, device=self.get_device()),
                               torch.tensor(trg_extended_token_ids, dtype=torch.int64, device=self.get_device()),
                               torch.tensor(trg_lengths, dtype=torch.int64, device=self.get_device()),
                               torch.tensor(np.array(invalid_copy_positions), dtype=torch.uint8, device=self.get_device()),
                               inp_str_reps, inp_ids, torch.tensor(np.array(code_features), dtype=torch.float32, device=self.get_device()),
                               torch.tensor(np.array(nl_features), dtype=torch.float32, device=self.get_device()),
                               torch.tensor(np.array(ast_features), dtype=torch.float32, device=self.get_device()))
            )
        return batches
    
    def get_encoder_output(self, batch_data):
        code_embedded_tokens = self.code_features_to_embedding(torch.cat(
            [self.embedding_store.get_code_embeddings(batch_data.code_ids), batch_data.code_features], dim=-1))
        code_hidden_states, code_final_state = self.code_encoder.forward(code_embedded_tokens,
            batch_data.code_lengths, self.get_device())
        old_nl_embedded_tokens = self.nl_features_to_embedding(torch.cat(
            [self.embedding_store.get_nl_embeddings(batch_data.old_nl_ids), batch_data.nl_features], dim=-1))
        old_nl_hidden_states, old_nl_final_state = self.nl_encoder.forward(old_nl_embedded_tokens,
            batch_data.old_nl_lengths, self.get_device())
        ast_embedded_tokens = self.ast_features_to_embedding(torch.cat(
            [self.embedding_store.get_code_embeddings(batch_data.ast_ids), batch_data.ast_features], dim=-1))
        ast_hidden_states, ast_final_state = self.ast_encoder.forward(ast_embedded_tokens,
            batch_data.ast_lengths, self.get_device())

        encoder_hidden_states, input_lengths = merge_3_encoder_outputs(code_hidden_states, batch_data.code_lengths,
                                                                       old_nl_hidden_states, batch_data.old_nl_lengths,
                                                                       ast_hidden_states, batch_data.ast_lengths,
                                                                       self.get_device())

        encoder_final_state = torch.einsum('bd,dh->bh',
            torch.cat([code_final_state, old_nl_final_state, ast_final_state], dim=-1),
            self.encoder_final_to_decoder_initial)
        mask = (torch.arange(encoder_hidden_states.shape[1], device=self.get_device()).view(1, -1) >=
                input_lengths.view(-1, 1)).unsqueeze(1)
        
        code_masks = (torch.arange(code_hidden_states.shape[1], device=self.get_device()).view(1, -1) >=
                      batch_data.code_lengths.view(-1, 1)).unsqueeze(1)
        old_nl_masks = (torch.arange(old_nl_hidden_states.shape[1], device=self.get_device()).view(1, -1) >=
                        batch_data.old_nl_lengths.view(-1, 1)).unsqueeze(1)
        ast_masks = (torch.arange(ast_hidden_states.shape[1], device=self.get_device()).view(1, -1) >=
                     batch_data.ast_lengths.view(-1, 1)).unsqueeze(1)
        
        return encoder_hidden_states, encoder_final_state, mask, code_hidden_states, old_nl_hidden_states, ast_hidden_states, code_masks, old_nl_masks, ast_masks
    
    def forward(self, batch_data):
        """Computes the loss against the gold sequences corresponding to the examples in the batch. NOTE: teacher-forcing."""
        encoder_hidden_states, initial_state, inp_length_mask, code_hidden_states, old_nl_hidden_states, ast_hidden_states, \
        code_masks, old_nl_masks, ast_masks = self.get_encoder_output(batch_data)

        decoder_input_embeddings = self.embedding_store.get_nl_embeddings(batch_data.trg_nl_ids)[:, :-1]
        decoder_states, decoder_final_state, generation_logprobs, copy_logprobs = self.decoder.forward(initial_state, decoder_input_embeddings,
            encoder_hidden_states, code_hidden_states, old_nl_hidden_states, ast_hidden_states, inp_length_mask, code_masks, old_nl_masks, ast_masks)
        gold_generation_ids = batch_data.trg_nl_ids[:, 1:].unsqueeze(-1)
        gold_generation_logprobs = torch.gather(input=generation_logprobs, dim=-1,
                                                index=gold_generation_ids).squeeze(-1)
        copy_logprobs = copy_logprobs.masked_fill(
            batch_data.invalid_copy_positions[:, 1:, :encoder_hidden_states.shape[1]].bool(), float('-inf'))
        gold_copy_logprobs = copy_logprobs.logsumexp(dim=-1)
        gold_logprobs = torch.logsumexp(torch.cat(
            [gold_generation_logprobs.unsqueeze(-1), gold_copy_logprobs.unsqueeze(-1)], dim=-1), dim=-1)
        gold_logprobs = gold_logprobs.masked_fill(torch.arange(batch_data.trg_nl_ids[:, 1:].shape[-1],
            device=self.get_device()).unsqueeze(0) >= batch_data.trg_nl_lengths.unsqueeze(-1)-1, 0)
        
        likelihood_by_example = gold_logprobs.sum(dim=-1)

        likelihood_by_example = likelihood_by_example/(batch_data.trg_nl_lengths-1).float() 
        
        return -(likelihood_by_example).mean()
    
    def beam_decode(self, batch_data):
        encoder_hidden_states, initial_state, inp_length_mask, code_hidden_states, old_nl_hidden_states, \
        ast_hidden_states, code_masks, old_nl_masks, ast_masks = self.get_encoder_output(batch_data)
        predictions, scores = self.decoder.beam_decode(initial_state, encoder_hidden_states, code_hidden_states,
                              old_nl_hidden_states, ast_hidden_states, inp_length_mask, self.max_nl_length, batch_data,
                              code_masks, old_nl_masks, ast_masks, self.get_device())
        
        decoded_output = []
        batch_size = initial_state.shape[0]

        for i in range(batch_size):
            beam_output = []
            for j in range(len(predictions[i])):
                token_ids = predictions[i][j]
                tokens = self.embedding_store.get_nl_tokens(token_ids, batch_data.input_ids[i],
                    batch_data.input_str_reps[i])
                beam_output.append((tokens, scores[i][j]))
            decoded_output.append(beam_output)
        return decoded_output
    
    def get_device(self):
        if self.torch_device_name == 'gpu':
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def run_gradient_step(self, batch_data):
        self.optimizer.zero_grad()
        loss = self.forward(batch_data)
        loss.backward()
        self.optimizer.step()
        return float(loss.cpu())

    def run_train(self, train_data, valid_data):
        print("Start getting valid batches...")
        valid_batches = self.get_batches(valid_data)
        print("Start getting train batches...")
        train_batches = self.get_batches(train_data, shuffle=True)
        print("********  Start training  ********")

        best_loss = float('inf')
        patience_tally = 0

        step = 0
        for epoch in range(MAX_EPOCHS):
            if patience_tally > PATIENCE:
                print('-----------------Stop.')
                break
            
            self.train()
            random.shuffle(train_batches)
            
            train_loss = 0
            for batch_data in tqdm(train_batches):
                cur_batch_loss = self.run_gradient_step(batch_data)
                train_loss += cur_batch_loss
                step += 1
            self.eval()
            validation_loss = 0
            with torch.no_grad():
                for batch_data in valid_batches:
                    validation_loss += float(self.forward(batch_data).cpu())

            validation_loss = validation_loss/len(valid_batches)

            if validation_loss <= best_loss:
                torch.save(self, self.model_path)
                saved = True
                best_loss = validation_loss
                patience_tally = 0
            else:
                saved = False
                patience_tally += 1
            
            print('Epoch: {}'.format(epoch))
            print('Training loss: {}'.format(train_loss/len(train_batches)))
            print('Validation loss: {}'.format(validation_loss))
            if saved:
                print('Saved')
            print('-----------------------------------')

    def get_likelihood_scores(self, comment_generation_model, formatted_beam_predictions, test_example):
        batch_examples = []
        for j in range(len(formatted_beam_predictions)):
            batch_examples.append(Example(test_example.id, test_example.old_comment, test_example.old_comment_tokens,
                ' '.join(formatted_beam_predictions[j]), formatted_beam_predictions[j],
                test_example.old_code, test_example.old_code_tokens, test_example.new_code,
                test_example.new_code_tokens))
        
        batch_data = comment_generation_model.get_batches(batch_examples)[0]
        return np.asarray(comment_generation_model.compute_generation_likelihood(batch_data))

    def run_evaluation(self, test_data):
        self.eval()
        print("Start getting test batches...")
        test_batches = self.get_batches(test_data)
        test_predictions = []

        gold_strs = []
        pred_strs = []
        src_strs = []

        references = []
        pred_instances = []

        with torch.no_grad():
            for b_idx, batch_data in enumerate(tqdm(test_batches)):
                test_predictions.extend(self.beam_decode(batch_data))

        top_k_test_predictions = []
        for pred in test_predictions:
            top_k_test_predictions.append([tup[0] for tup in pred[0:5]])
        test_predictions = top_k_test_predictions

        for i in range(len(test_predictions)):
            cur_preds = []
            for candidate_pre in test_predictions[i]:
                pred_str = diff_utils.format_minimal_diff_spans(test_data[i].old_comment_tokens, candidate_pre)
                cur_preds.append(pred_str)
            prediction = cur_preds[0].split()
            gold_str = test_data[i].new_comment

            gold_strs.append(gold_str)
            pred_strs.append(cur_preds)
            src_strs.append(test_data[i].old_comment)

            references.append([test_data[i].new_comment_tokens])
            pred_instances.append(prediction)

        write_predictions(pred_strs, 'pred.txt')
        write_reference(src_strs, 'src.txt')
        write_reference(gold_strs, 'ref.txt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str, default='dataset/')
    parser.add_argument('-model_path', type=str, default='models/model_1.pt')
    parser.add_argument('--test_mode', default=True, action='store_true')
    args = parser.parse_args()

    if args.test_mode:
        print("********  Test Mode  ********")
        test_examples = read_examples_from_file(os.path.join(args.data_path, 'test.jsonl'))
        cuda_flag = False
        if cuda_flag:
            print("==========use cuda==========")
            model = torch.load(args.model_path)
            model.torch_device_name = 'gpu'
            model.cuda()
            for c in model.children():
                c.cuda()
        else:
            print("==========use cpu==========")
            model = torch.load(args.model_path, map_location=torch.device('cpu'))
            model.torch_device_name = 'cpu'
            model.cpu()
            for c in model.children():
                c.cpu()
        model.run_evaluation(test_examples)

    else:
        print("********  Train Mode  ********")
        print("Start reading examples...")
        train_examples = read_examples_from_file(os.path.join(args.data_path, 'train.jsonl'))
        valid_examples = read_examples_from_file(os.path.join(args.data_path, 'valid.jsonl'))
        print("Initialize the model...")
        model = CommentUpdateModel(args.model_path)
        model.initialize(train_examples)
        if torch.cuda.is_available():
            print("==========use cuda==========")
            model.torch_device_name = 'gpu'
            model.cuda()
            for c in model.children():
                c.cuda()
        else:
            print("==========use cpu==========")
            model.torch_device_name = 'cpu'
            model.cpu()
            for c in model.children():
                c.cpu()
        model.run_train(train_examples, valid_examples)
