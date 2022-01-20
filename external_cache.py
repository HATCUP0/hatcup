import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import numpy as np
from utils.diff_utils import *


with open('dataset/method_details.json') as f:
    method_details = json.load(f)

with open('dataset/tokenization_features.json') as f:
    tokenization_features = json.load(f)

stop_words = set(stopwords.words('english'))
java_keywords = set(['abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class',
         'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 'finally',
         'float', 'for', 'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long',
         'native', 'new', 'null', 'package', 'private', 'protected', 'public', 'return', 'short',
         'static', 'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient',
         'try', 'void', 'volatile', 'while'])

tags = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT',
        'POS','PRP','PRP$','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB',
        'OTHER']

AST_OPERATION = ['add', 'delete', 'move', 'update']

AST_TYPES = ['ASSIGNMENT_OPERATOR', 'AnonymousClassDeclaration', 'ArrayAccess', 'ArrayCreation', 'ArrayInitializer',
             'ArrayType', 'AssertStatement', 'Assignment', 'Block', 'BooleanLiteral', 'BreakStatement',
             'CastExpression', 'CatchClause', 'CharacterLiteral', 'ClassInstanceCreation', 'ConditionalExpression',
             'ContinueStatement', 'CreationReference', 'Dimension', 'DoStatement', 'EmptyStatement',
             'EnhancedForStatement', 'ExpressionMethodReference', 'ExpressionStatement', 'FieldAccess',
             'FieldDeclaration', 'ForStatement', 'INFIX_EXPRESSION_OPERATOR', 'IfStatement', 'InfixExpression',
             'Initializer', 'InstanceofExpression', 'Javadoc', 'LabeledStatement', 'LambdaExpression',
             'METHOD_INVOCATION_ARGUMENTS', 'METHOD_INVOCATION_RECEIVER', 'MarkerAnnotation', 'MemberValuePair',
             'MethodDeclaration', 'MethodInvocation', 'Modifier', 'NormalAnnotation', 'NullLiteral',
             'NumberLiteral', 'POSTFIX_EXPRESSION_OPERATOR', 'PREFIX_EXPRESSION_OPERATOR', 'ParameterizedType',
             'ParenthesizedExpression', 'PostfixExpression', 'PrefixExpression', 'PrimitiveType', 'QualifiedName',
             'ReturnStatement', 'SimpleName', 'SimpleType', 'SingleMemberAnnotation', 'SingleVariableDeclaration',
             'StringLiteral', 'SuperMethodInvocation', 'SwitchCase', 'SwitchStatement', 'SynchronizedStatement',
             'TYPE_DECLARATION_KIND', 'TagElement', 'TextElement', 'ThisExpression', 'ThrowStatement',
             'TryStatement', 'TypeDeclaration', 'TypeDeclarationStatement', 'TypeLiteral', 'TypeParameter',
             'UnionType', 'VARARGS_TYPE', 'VariableDeclarationExpression', 'VariableDeclarationFragment',
             'VariableDeclarationStatement', 'WhileStatement', 'WildcardType', 'OTHER']

NUM_CODE_FEATURES = 15
NUM_NL_FEATURES = 17 + len(tags)
NUM_AST_FEATURES = 10

def is_java_keyword(token):
    return token in java_keywords

def is_operator(token):
    for s in token:
        if s.isalnum():
            return False
    return True

def get_return_type_subtokens(example):
    return method_details[example.id]['return_type_subtokens']

def get_old_return_type_subtokens(example):
    return method_details[example.id]['old_return_type_subtokens']

def get_method_name_subtokens(example):
    return method_details[example.id]['method_name_subtokens']

def get_new_return_sequence(example):
    return method_details[example.id]['new_return_sequence']

def get_old_return_sequence(example):
    return method_details[example.id]['old_return_sequence']

def get_old_code(example):
    return method_details[example.id]['old_code']

def get_new_code(example):
    return method_details[example.id]['new_code']

def get_edit_span_subtoken_tokenization_labels(example):
    return tokenization_features[example.id]['edit_span_subtoken_labels']

def get_edit_span_subtoken_tokenization_indices(example):
    return tokenization_features[example.id]['edit_span_subtoken_indices']

def get_nl_subtoken_tokenization_labels(example):
    return tokenization_features[example.id]['old_nl_subtoken_labels']

def get_nl_subtoken_tokenization_indices(example):
    return tokenization_features[example.id]['old_nl_subtoken_indices']

def get_ast_subtoken_tokenization_labels(example):
    return tokenization_features[example.id]['ast_subtoken_labels']

def get_ast_subtoken_tokenization_indices(example):
    return tokenization_features[example.id]['ast_subtoken_indices']

def get_code_features(code_sequence, example, max_code_length):
    old_return_type_subtokens = get_old_return_type_subtokens(example)
    new_return_type_subtokens = get_return_type_subtokens(example)
    method_name_subtokens = get_method_name_subtokens(example)

    old_return_sequence = get_old_return_sequence(example)
    new_return_sequence = get_new_return_sequence(example)

    old_return_line_terms = set([t for t in old_return_sequence if not is_java_keyword(t) and not is_operator(t)])
    new_return_line_terms = set([t for t in new_return_sequence if not is_java_keyword(t) and not is_operator(t)])
    return_line_intersection = old_return_line_terms.intersection(new_return_line_terms)

    old_set = set(old_return_type_subtokens)
    new_set = set(new_return_type_subtokens)

    intersection = old_set.intersection(new_set)

    features = np.zeros((max_code_length, NUM_CODE_FEATURES), dtype=np.bool)


    last_command = None

    edit_span_subtoken_labels = get_edit_span_subtoken_tokenization_labels(example)
    edit_span_subtoken_indices = get_edit_span_subtoken_tokenization_indices(example)

    for i, token in enumerate(code_sequence):
        if i >= max_code_length:
            break
        if token in intersection:
            features[i][0] = True
        elif token in old_set:
            features[i][1] = True
        elif token in new_set:
            features[i][2] = True
        else:
            features[i][3] = True

        if token in return_line_intersection:
            features[i][4] = True
        elif token in old_return_line_terms:
            features[i][5] = True
        elif token in new_return_line_terms:
            features[i][6] = True
        else:
            features[i][7] = True

        if not is_edit_keyword(token):
            if last_command == KEEP:
                features[i][8] = 1
            elif last_command == INSERT:
                features[i][9] = 1
            elif last_command == DEL:
                features[i][10] = 1
            elif last_command == UPDATEFROM:
                features[i][11] = 1
            else:
                features[i][12] = 1
        else:
            last_command = token
        
        features[i][13] = edit_span_subtoken_labels[i]
        features[i][14] = edit_span_subtoken_indices[i]

    return features.astype(np.float32)

def get_nl_features(old_nl_sequence, example, max_nl_length):
    ast_diff = example.ast_diff

    frequency_map = dict()
    for tok in old_nl_sequence:
        if tok not in frequency_map:
            frequency_map[tok] = 0
        frequency_map[tok] += 1
    
    pos_tags = pos_tag(word_tokenize(' '.join(old_nl_sequence)))
    pos_tag_indices = []
    for _, t in pos_tags:
        if t in tags:
            pos_tag_indices.append(tags.index(t))
        else:
            pos_tag_indices.append(tags.index('OTHER'))

    
    old_return_type_subtokens = get_old_return_type_subtokens(example)
    new_return_type_subtokens = get_return_type_subtokens(example)

    old_return_sequence = get_old_return_sequence(example)
    new_return_sequence = get_new_return_sequence(example)

    old_return_line_terms = set([t for t in old_return_sequence if not is_java_keyword(t) and not is_operator(t)])
    new_return_line_terms = set([t for t in new_return_sequence if not is_java_keyword(t) and not is_operator(t)])
    return_line_intersection = old_return_line_terms.intersection(new_return_line_terms)

    old_set = set(old_return_type_subtokens)
    new_set = set(new_return_type_subtokens)

    intersection = old_set.intersection(new_set)

    method_name_subtokens = get_method_name_subtokens(example)

    nl_subtoken_labels = get_nl_subtoken_tokenization_labels(example)
    nl_subtoken_indices = get_nl_subtoken_tokenization_indices(example)
        
    features = np.zeros((max_nl_length, NUM_NL_FEATURES), dtype=np.bool)
    for i in range(len(old_nl_sequence)):
        if i >= max_nl_length:
            break
        token = old_nl_sequence[i].lower()
        if token in intersection:
            features[i][0] = True
        elif token in old_set:
            features[i][1] = True
        elif token in new_set:
            features[i][2] = True
        else:
            features[i][3] = True

        if token in return_line_intersection:
            features[i][4] = True
        elif token in old_return_line_terms:
            features[i][5] = True
        elif token in new_return_line_terms:
            features[i][6] = True
        else:
            features[i][7] = True

        if token in ast_diff['add']:
            features[i][8] = True
        elif token in ast_diff['delete']:
            features[i][9] = True
        elif token in ast_diff['move']:
            features[i][10] = True
        elif token in ast_diff['update']:
            features[i][11] = True
        else:
            features[i][12] = True
        features[i][13] = token in stop_words
        features[i][14] = frequency_map[token] > 1
        features[i][15] = nl_subtoken_labels[i]
        features[i][16] = nl_subtoken_indices[i]
        features[i][17 + pos_tag_indices[i]] = 1

    return features.astype(np.float32)


def get_ast_features(ast_diff_tokens, example, max_ast_length):
    ast_diff = example.ast_diff
    ast_subtoken_labels = get_ast_subtoken_tokenization_labels(example)
    ast_subtoken_indices = get_ast_subtoken_tokenization_indices(example)
    old_nl_tokens = set(example.old_comment_tokens)
    features = np.zeros((max_ast_length, NUM_AST_FEATURES), dtype=np.bool)
    for i in range(len(ast_diff_tokens)):
        if i >= NUM_AST_FEATURES:
            break
        cur_token = ast_diff_tokens[i]
        features[i][0] = cur_token in ast_diff['add']
        features[i][1] = cur_token in ast_diff['delete']
        features[i][2] = cur_token in ast_diff['move']
        features[i][3] = cur_token in ast_diff['update']

        if is_edit_keyword(cur_token):
            features[i][4] = True
        if is_java_keyword(cur_token):
            features[i][5] = True
        if is_operator(cur_token):
            features[i][6] = True
        if cur_token in old_nl_tokens:
            features[i][7] = True

        features[i][8] = ast_subtoken_labels[i]
        features[i][9] = ast_subtoken_indices[i]

    return features.astype(np.float32)

