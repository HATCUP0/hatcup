import difflib

UPDATE = '<UPDATE>'
UPDATEFROM = '<UPDATEFROM>'
UPDATETO = '<UPDATETO>'
UPDATE_END = '<UPDATE_END>'
UPDATEFROM_KEEP_BEFORE = '<UPDATEFROM_KEEP_BEFORE>'
UPDATETO_KEEP_BEFORE = '<UPDATETO_KEEP_BEFORE>'
UPDATEFROM_KEEP_AFTER = '<UPDATEFROM_KEEP_AFTER>'
UPDATETO_KEEP_AFTER = '<UPDATETO_KEEP_AFTER>'
UPDATEFROM_DEL_KEEP_BEFORE = '<UPDATEFROM_DEL_KEEP_BEFORE>'
UPDATETO_DEL_KEEP_BEFORE = '<UPDATETO_DEL_KEEP_BEFORE>'
UPDATEFROM_DEL_KEEP_AFTER = '<UPDATEFROM_DEL_KEEP_AFTER>'
UPDATETO_DEL_KEEP_AFTER = '<UPDATETO_DEL_KEEP_AFTER>'

INSERT = '<INSERT>'
INSERT_OLD = '<INSERT_OLD>'
INSERT_NEW = '<INSERT_NEW>'
INSERT_END = '<INSERT_END>'
INSERT_OLD_KEEP_BEFORE = '<INSERT_OLD_KEEP_BEFORE>'
INSERT_NEW_KEEP_BEFORE = '<INSERT_NEW_KEEP_BEFORE>'
INSERT_OLD_KEEP_AFTER = '<INSERT_OLD_KEEP_AFTER>'
INSERT_NEW_KEEP_AFTER = '<INSERT_NEW_KEEP_AFTER>'

DEL = '<DEL>'
DEL_END = '<DEL_END>'

KEEP = '<KEEP>'
KEEP_END = '<KEEP_END>'

class EditNode:
    def __init__(self, edit_type, children, prev, next):
        self.edit_type = edit_type
        self.children = children
        self.prev = prev
        self.next = next

def get_edit_keywords():
    return [UPDATE, UPDATEFROM, UPDATETO, UPDATE_END, UPDATEFROM_KEEP_BEFORE, UPDATETO_KEEP_BEFORE, UPDATEFROM_KEEP_AFTER,
        UPDATETO_KEEP_AFTER, UPDATEFROM_DEL_KEEP_BEFORE, UPDATETO_DEL_KEEP_BEFORE, UPDATEFROM_DEL_KEEP_AFTER,
        UPDATETO_DEL_KEEP_AFTER, INSERT, INSERT_OLD, INSERT_NEW, INSERT_END, INSERT_OLD_KEEP_BEFORE, INSERT_NEW_KEEP_BEFORE,
        INSERT_OLD_KEEP_AFTER, INSERT_NEW_KEEP_AFTER, DEL, DEL_END, KEEP, KEEP_END]

def get_index(search_tokens, full_tokens):
    if len(search_tokens) == 0:
        return 0

    possible_positions = [k for k in range(len(full_tokens)) if full_tokens[k] == search_tokens[0]]
    if len(possible_positions) == 0:
        return -1
    
    if len(possible_positions) == 1:
        return possible_positions[0]
    
    for p in possible_positions:
        s_pos = 1
        f_pos = p + 1
        invalid = False

        while s_pos < len(search_tokens) and f_pos < len(full_tokens):
            if search_tokens[s_pos] != full_tokens[f_pos]:
                invalid = True
                break
            
            s_pos += 1
            f_pos += 1
        
        if not invalid:
            return p
    return -1

def get_valid_positions(search_str, full_str):
    search_sequence = search_str.split()
    full_sequence = full_str.split()

    if len(search_sequence) == 0:
        return 0

    possible_positions = [p for p in range(len(full_sequence)) if full_sequence[p] == search_sequence[0]]
    valid_positions = []

    for p in possible_positions:
        valid = True
        for i in range(len(search_sequence)):
            if p+i >= len(full_sequence) or full_sequence[p+i] != search_sequence[i]:
                valid = False
                break
        if valid:
            valid_positions.append(p)

    return valid_positions

def get_frequency(search_str, full_str):
    return len(get_valid_positions(search_str, full_str))

def get_coarse_diff_structure(old_tokens, new_tokens):
    nodes = []
    last_node = None
    for edit_type, o_start, o_end, n_start, n_end in difflib.SequenceMatcher(None, old_tokens, new_tokens).get_opcodes():
        if edit_type == 'equal':
            edit_node = EditNode(KEEP, old_tokens[o_start:o_end], last_node, None)
        elif edit_type == 'replace':
            edit_node = EditNode(UPDATE, old_tokens[o_start:o_end] + [UPDATETO] + new_tokens[n_start:n_end], last_node, None)
        elif edit_type == 'insert':
            edit_node = EditNode(INSERT, new_tokens[n_start:n_end], last_node, None)
        else:
            edit_node = EditNode(DEL, old_tokens[o_start:o_end], last_node, None)
        
        if last_node:
            last_node.next = edit_node
        last_node = edit_node
        nodes.append(edit_node)
    return nodes

def merge_diff_actions(diff_structure):
    mega_nodes = []
    curr_mega_node = []
    for node in diff_structure:
        if len(node.children) == 1:
            curr_mega_node.append(node)
        else:
            if len(curr_mega_node) == 1:
                curr_mega_node.append(node)
                mega_nodes.append(curr_mega_node)
                curr_mega_node = []
            else:
                if len(curr_mega_node) > 0:
                    mega_nodes.append(curr_mega_node)
                    curr_mega_node = []
                mega_nodes.append([node])
    
    if len(curr_mega_node) == 1:
        mega_nodes[-1].extend(curr_mega_node)
    elif len(curr_mega_node) > 0:
        mega_nodes.append(curr_mega_node)
    
    new_nodes = []
    for m_node in mega_nodes:
        if len(m_node) == 1:
            new_nodes.append(m_node[0])
            continue
        
        old_tokens = []
        new_tokens = []
        for sub in m_node:
            if sub.edit_type == KEEP:
                old_tokens.extend(sub.children)
                new_tokens.extend(sub.children)
            elif sub.edit_type == INSERT:
                new_tokens.extend(sub.children)
            elif sub.edit_type == DEL:
                old_tokens.extend(sub.children)
            else:
                rep_idx = sub.children.index(UPDATETO)
                old_tokens.extend(sub.children[:rep_idx])
                new_tokens.extend(sub.children[rep_idx+1:])
        
        update_node = EditNode(UPDATE, old_tokens + [UPDATETO] + new_tokens, None, None)
        new_nodes.append(update_node)
    
    n = 0
    final_new_nodes = []
    while n < len(new_nodes):
        while n < len(new_nodes) and new_nodes[n].edit_type not in [INSERT, UPDATE, DEL]:
            final_new_nodes.append(new_nodes[n])
            n += 1

        to_merge = []
        while n < len(new_nodes) and new_nodes[n].edit_type in [INSERT, UPDATE, DEL]:
            to_merge.append(new_nodes[n])
            n += 1
        
        if len(to_merge) > 0:
            old_tokens = []
            new_tokens = []
            for node in to_merge:
                if node.edit_type == INSERT:
                    new_tokens.extend(node.children)
                elif node.edit_type == DEL:
                    old_tokens.extend(node.children)
                elif node.edit_type == UPDATE:
                    rep_idx = node.children.index(UPDATETO)
                    old_tokens.extend(node.children[:rep_idx])
                    new_tokens.extend(node.children[rep_idx+1:])
            
            update_node = EditNode(UPDATE, old_tokens + [UPDATETO] + new_tokens, None, None)
            final_new_nodes.append(update_node)
        
        if n < len(new_nodes):
            final_new_nodes.append(new_nodes[n])
        
        n += 1
    
    new_nodes = final_new_nodes
    for n, node in enumerate(new_nodes):
        if n > 0:
            new_nodes[n].next = node
            node.prev = new_nodes[n]
        if n+1 < len(new_nodes):
            new_nodes[n+1].prev = node
            node.next = new_nodes[n+1]
    
    return new_nodes

def compute_code_diffs(old_tokens, new_tokens):
    spans = []
    tokens = []
    commands = []

    for edit_type, o_start, o_end, n_start, n_end in difflib.SequenceMatcher(None, old_tokens, new_tokens).get_opcodes():
        if edit_type == 'equal':
            spans.extend([KEEP] + old_tokens[o_start:o_end] + [KEEP_END])
            for i in range(o_start, o_end):
                tokens.extend([KEEP, old_tokens[i]])
                commands.append(KEEP)
        elif edit_type == 'replace':
            spans.extend([UPDATEFROM] + old_tokens[o_start:o_end] + [UPDATETO] + new_tokens[n_start:n_end] + [UPDATE_END])
            for i in range(o_start, o_end):
                tokens.extend([UPDATEFROM, old_tokens[i]])
                commands.append(UPDATEFROM)
            for j in range(n_start, n_end):
                tokens.extend([UPDATETO, new_tokens[j]])
                commands.extend([UPDATETO, new_tokens[j]])
        elif edit_type == 'insert':
            spans.extend([INSERT] + new_tokens[n_start:n_end] + [INSERT_END])
            for j in range(n_start, n_end):
                tokens.extend([INSERT, new_tokens[j]])
                commands.extend([INSERT, new_tokens[j]])
        else:
            spans.extend([DEL] + old_tokens[o_start:o_end] + [DEL_END])
            for i in range(o_start, o_end):
                tokens.extend([DEL, old_tokens[i]])
                commands.append(DEL)

    return spans, tokens, commands

def compute_minimal_code_diffs(old_tokens, new_tokens):
    spans = []
    tokens = []
    commands = []

    for edit_type, o_start, o_end, n_start, n_end in difflib.SequenceMatcher(None, old_tokens, new_tokens).get_opcodes():
        if edit_type == 'equal':
            continue
        elif edit_type == 'replace':
            spans.extend([UPDATEFROM] + old_tokens[o_start:o_end] + [UPDATETO] + new_tokens[n_start:n_end] + [UPDATE_END])
            for i in range(o_start, o_end):
                tokens.extend([UPDATEFROM, old_tokens[i]])
                commands.append(UPDATEFROM)
            for j in range(n_start, n_end):
                tokens.extend([UPDATETO, new_tokens[j]])
                commands.extend([UPDATETO, new_tokens[j]])
        elif edit_type == 'insert':
            spans.extend([INSERT] + new_tokens[n_start:n_end] + [INSERT_END])
            for j in range(n_start, n_end):
                tokens.extend([INSERT, new_tokens[j]])
                commands.extend([INSERT, new_tokens[j]])
        else:
            spans.extend([DEL] + old_tokens[o_start:o_end] + [DEL_END])
            for i in range(o_start, o_end):
                tokens.extend([DEL, old_tokens[i]])
                commands.append(DEL)

    return spans, tokens, commands

def compute_comment_diffs(old_tokens, new_tokens):
    spans = []
    tokens = []
    commands = []

    diff_nodes = get_coarse_diff_structure(old_tokens, new_tokens)
    diff_nodes = merge_diff_actions(diff_nodes)

    for node in diff_nodes:
        if node.edit_type == KEEP:
            spans.extend([KEEP] + node.children + [KEEP_END])
            for i in range(len(node.children)):
                tokens.extend([KEEP, node.children[i]])
                commands.append(KEEP)
        elif node.edit_type == UPDATE:
            o_end = node.children.index(UPDATETO)
            n_start = o_end + 1
            n_end = len(node.children)
            spans.extend([UPDATEFROM] + node.children + [UPDATE_END])
            for i in range(o_end):
                tokens.extend([UPDATEFROM, node.children[i]])
                commands.append(UPDATEFROM)
            for j in range(n_start, n_end):
                tokens.extend([UPDATETO, node.children[j]])
                commands.extend([UPDATETO, node.children[j]])
        elif node.edit_type == INSERT:
            spans.extend([INSERT] + node.children + [INSERT_END])
            for j in range(len(node.children)):
                tokens.extend([INSERT, node.children[j]])
                commands.extend([INSERT, node.children[j]])
        else:
            spans.extend([DEL] + node.children + [DEL_END])
            for i in range(len(node.children)):
                tokens.extend([DEL, node.children[i]])
                commands.append(DEL)

    return spans, tokens, commands

def compute_minimal_comment_diffs(old_tokens, new_tokens):
    spans = []
    tokens = []
    commands = []

    old_str = ' '.join(old_tokens)
    diff_nodes = get_coarse_diff_structure(old_tokens, new_tokens)
    
    new_nodes = []
    
    for n, node in enumerate(diff_nodes):
        if node.edit_type == KEEP:
            new_nodes.append(node)
        
        elif node.edit_type == DEL:
            search_str = ' '.join(node.children)
            if get_frequency(search_str, old_str) == 1:
                node.children.insert(0, DEL)
                new_nodes.append(node)
                continue
            
            if node.prev and node.prev.edit_type == KEEP:
                adopted_children = []
                found_substring = False
                while not found_substring and len(node.prev.children) > 0:
                    adopted_children.insert(0, node.prev.children.pop())
                    search_str = ' '.join(adopted_children + node.children)
                    found_substring = get_frequency(search_str, old_str) == 1
                
                if found_substring:
                    new_children = [UPDATEFROM_DEL_KEEP_BEFORE] + adopted_children + node.children + [UPDATETO_DEL_KEEP_BEFORE] + adopted_children 
                    new_node = EditNode(UPDATE, new_children, node.prev, node.next)
                    node.prev.next = new_node
                    if node.next:
                        node.next.prev = new_node
                    new_nodes.append(new_node)
                    continue
                else:
                    node.prev.children.extend(adopted_children)
            
            if node.next and node.next.edit_type == KEEP:
                adopted_children = []
                found_substring = False
                while not found_substring and len(node.next.children) > 0:
                    adopted_children.append(node.next.children.pop(0))
                    search_str = ' '.join(node.children + adopted_children)
                    found_substring = get_frequency(search_str, old_str) == 1
                
                if found_substring:
                    new_children = [UPDATEFROM_DEL_KEEP_AFTER] + node.children + adopted_children + [UPDATETO_DEL_KEEP_AFTER] + adopted_children
                    new_node = EditNode(UPDATE, new_children, node.prev, node.next)

                    if node.prev:
                        node.prev.next = new_node
                    
                    node.next.prev = new_node
                    new_nodes.append(new_node)
                    continue
                else:
                    node.next.children = adopted_children + node.next.children
            
            return get_full_update_span(old_tokens, new_tokens), tokens, commands
        
        elif node.edit_type == UPDATE:
            rep_idx = node.children.index(UPDATETO)
            rep_old_children = node.children[:rep_idx]
            rep_new_children = node.children[rep_idx+1:]
            search_str = ' '.join(rep_old_children)
            
            if get_frequency(search_str, old_str) == 1:
                node.children.insert(0, UPDATEFROM)
                new_nodes.append(node)
                continue
            
            if node.prev and node.prev.edit_type == KEEP:
                adopted_children = []
                found_substring = False
                while not found_substring and len(node.prev.children) > 0:
                    adopted_children.insert(0, node.prev.children.pop())
                    search_str = ' '.join(adopted_children + rep_old_children)
                    found_substring = get_frequency(search_str, old_str) == 1
                
                if found_substring:
                    new_children = [UPDATEFROM_KEEP_BEFORE] + adopted_children + rep_old_children + [UPDATETO_KEEP_BEFORE] + adopted_children + rep_new_children
                    new_node = EditNode(UPDATE, new_children, node.prev, node.next)
                    node.prev.next = new_node
                    if node.next:
                        node.next.prev = new_node
                    new_nodes.append(new_node)
                    continue
                else:
                    node.prev.children.extend(adopted_children)
            
            if node.next and node.next.edit_type == KEEP:
                adopted_children = []
                found_substring = False
                while not found_substring and len(node.next.children) > 0:
                    adopted_children.append(node.next.children.pop(0))
                    search_str = ' '.join(rep_old_children + adopted_children)
                    found_substring = get_frequency(search_str, old_str) == 1
                
                if found_substring:
                    new_children = [UPDATEFROM_KEEP_AFTER] + rep_old_children + adopted_children + [UPDATETO_KEEP_AFTER] + rep_new_children + adopted_children
                    new_node = EditNode(UPDATE, new_children, node.prev, node.next)

                    if node.prev:
                        node.prev.next = new_node
                    
                    node.next.prev = new_node
                    new_nodes.append(new_node)
                    continue
                else:
                    node.next.children = adopted_children + node.next.children
            
            return get_full_update_span(old_tokens, new_tokens), tokens, commands

        elif node.edit_type == INSERT:
            if node.prev and node.prev.edit_type == KEEP:
                adopted_children = []
                found_substring = False
                while not found_substring and len(node.prev.children) > 0:
                    adopted_children.insert(0, node.prev.children.pop())
                    search_str = ' '.join(adopted_children)
                    found_substring = get_frequency(search_str, old_str) == 1
                
                if found_substring:
                    new_children = [INSERT_OLD_KEEP_BEFORE] + adopted_children + [INSERT_NEW_KEEP_BEFORE] + adopted_children + node.children
                    new_node = EditNode(INSERT, new_children, node.prev, node.next)
                    node.prev.next = new_node
                    if node.next:
                        node.next.prev = new_node
                    new_nodes.append(new_node)
                    continue
                else:
                    node.prev.children.extend(adopted_children)
            
            if node.next and node.next.edit_type == KEEP:
                adopted_children = []
                found_substring = False
                while not found_substring and len(node.next.children) > 0:
                    adopted_children.append(node.next.children.pop(0))
                    search_str = ' '.join(adopted_children)
                    found_substring = get_frequency(search_str, old_str) == 1
                
                if found_substring:
                    new_children = [INSERT_OLD_KEEP_AFTER] + adopted_children + [INSERT_NEW_KEEP_AFTER] + node.children + adopted_children
                    new_node = EditNode(INSERT, new_children, node.prev, node.next)

                    if node.prev:
                        node.prev.next = new_node
                    
                    node.next.prev = new_node
                    new_nodes.append(new_node)
                    continue
                else:
                    node.next.children = adopted_children + node.next.children
            
            return get_full_update_span(old_tokens, new_tokens), tokens, commands
    
    for node in new_nodes:
        if 'INSERT' in node.edit_type:
            spans.extend(node.children + [INSERT_END])
        elif 'UPDATE' in node.edit_type:
            spans.extend(node.children + [UPDATE_END])
        elif 'DEL' in node.edit_type:
            spans.extend(node.children + [DEL_END])
    return spans, tokens, commands

def get_full_update_span(old_tokens, new_tokens):
    return [UPDATEFROM] + old_tokens + [UPDATETO] + new_tokens + [UPDATE_END]

def is_insert(token):
    return 'INSERT' in token

def is_keep(token):
    return 'KEEP' in token

def is_update(token):
    return 'UPDATE' in token

def is_delete(token):
    return 'DEL' in token

def is_insert_end(token):
    return is_insert(token) and is_end(token)

def is_insert_old(token):
    return is_insert(token) and 'OLD' in token

def is_insert_new(token):
    return is_insert(token) and 'NEW' in token

def is_keep_end(token):
    return is_keep(token) and is_end(token)

def is_update_end(token):
    return is_update(token) and is_end(token)

def is_update_old(token):
    return is_update(token) and 'FROM' in token

def is_update_new(token):
    return is_update(token) and 'TO' in token

def is_delete_end(token):
    return is_delete(token) and is_end(token)

def is_edit_keyword(token):
    return is_insert(token) or is_keep(token) or is_update(token) or is_delete(token)

def is_start(token):
    return is_edit_keyword(token) and 'NEW' not in token and 'TO' not in token and not is_end(token)

def is_end(token):
    return is_edit_keyword(token) and 'END' in token

def is_new(token):
    return is_edit_keyword(token) and ('NEW' in token or 'TO' in token)

def get_location(search_tokens, reference_tokens):
    ref_str = ' '.join(reference_tokens)
    for i in range(len(search_tokens)):
        for j in range(len(search_tokens), i, -1):
            search_str = ' '.join(search_tokens[i:j])
            valid_positions = get_valid_positions(search_str, ref_str)
            if len(valid_positions) > 0:
                return valid_positions[0], i, len(valid_positions) > 1
    return -1, -1, False

def format_minimal_diff_spans(reference_tokens, diff_span_tokens):
    ptr = 0
    new_comment_tokens = []

    post_delete = []
    post_update = []
    
    i = 0
    while i < len(diff_span_tokens):
        token = diff_span_tokens[i]

        if not is_start(token):
            i += 1
            continue
        
        if is_delete(token):
            j = i + 1
            delete_tokens = []
            multiple_delete = False

            while j < len(diff_span_tokens) and not is_delete_end(diff_span_tokens[j]):
                delete_tokens.append(diff_span_tokens[j])
                j += 1
            
            idx, d_start, multiple_delete = get_location(delete_tokens, reference_tokens[ptr:])
            
            if multiple_delete:
                post_delete.append(delete_tokens)

            if idx >= 0:
                before_match = delete_tokens[:d_start]
                for r in range(ptr, ptr+idx):
                    if reference_tokens[r] in before_match:
                        before_match.pop(before_match.index(reference_tokens[r]))
                    else:
                        new_comment_tokens.append(reference_tokens[r])
                
                ptr += idx
                remaining_delete_tokens = delete_tokens[d_start:]
                for d in remaining_delete_tokens:
                    if ptr < len(reference_tokens) and d in reference_tokens[ptr:]:
                        idx = reference_tokens[ptr:].index(d)
                        new_comment_tokens.extend(reference_tokens[ptr:ptr+idx])
                        ptr += idx + 1
        
        elif is_insert_old(token):
            j = i + 1
            delete_tokens = []
            insert_tokens = []
            multiple_insert = False
            
            while j < len(diff_span_tokens) and not is_insert_new(diff_span_tokens[j]):
                delete_tokens.append(diff_span_tokens[j])
                j += 1
            
            can_add = False
            idx, d_start, multiple_insert = get_location(delete_tokens, reference_tokens[ptr:])

            if idx >= 0:
                can_add = True
                before_match = delete_tokens[:d_start]
                for r in range(ptr, ptr+idx):
                    if reference_tokens[r] in before_match:
                        before_match.pop(before_match.index(reference_tokens[r]))
                    else:
                        new_comment_tokens.append(reference_tokens[r])
                
                ptr += idx
                remaining_delete_tokens = delete_tokens[d_start:]
                for d in remaining_delete_tokens:
                    if ptr < len(reference_tokens) and d in reference_tokens[ptr:]:
                        idx = reference_tokens[ptr:].index(d)
                        new_comment_tokens.extend(reference_tokens[ptr:ptr+idx])
                        ptr += idx + 1

            j += 1
            while j < len(diff_span_tokens) and not is_insert_end(diff_span_tokens[j]):
                insert_tokens.append(diff_span_tokens[j])
                if can_add:
                    new_comment_tokens.append(diff_span_tokens[j])
                j += 1
            
            if multiple_insert:
                post_update.append((delete_tokens, insert_tokens))

        elif is_update_old(token):
            j = i + 1
            delete_tokens = []
            insert_tokens = []
            multiple_update = False

            while j < len(diff_span_tokens) and not is_update_new(diff_span_tokens[j]):
                delete_tokens.append(diff_span_tokens[j])
                j += 1
            
            can_add = False
            idx, d_start, multiple_update = get_location(delete_tokens, reference_tokens[ptr:])
            if idx >= 0:
                can_add = True
                before_match = delete_tokens[:d_start]
                for r in range(ptr, ptr+idx):
                    if reference_tokens[r] in before_match:
                        before_match.pop(before_match.index(reference_tokens[r]))
                    else:
                        new_comment_tokens.append(reference_tokens[r])
                
                ptr += idx
                remaining_delete_tokens = delete_tokens[d_start:]
                for d in remaining_delete_tokens:
                    if ptr < len(reference_tokens) and d in reference_tokens[ptr:]:
                        idx = reference_tokens[ptr:].index(d)
                        new_comment_tokens.extend(reference_tokens[ptr:ptr+idx])
                        ptr += idx + 1
            
            j += 1   
            while j < len(diff_span_tokens) and not is_update_end(diff_span_tokens[j]):
                insert_tokens.append(diff_span_tokens[j])
                if can_add:
                    new_comment_tokens.append(diff_span_tokens[j])
                j += 1
            
            if multiple_update:
                post_update.append((delete_tokens, insert_tokens))
        else:
            print("Error!!!!!!!!!!!")
            print("reference_tokens:", reference_tokens)
            print("diff_span_tokens:", diff_span_tokens)
            raise ValueError('Invalid: {}'.format(token))
        i = j+1

    if ptr < len(reference_tokens):
        new_comment_tokens.extend(reference_tokens[ptr:])
    
    if len(post_delete) > 0:
        delete_positions = []
        for d in post_delete:
            start_positions = get_valid_positions(' '.join(d), ' '.join(new_comment_tokens))
            for s in start_positions:
                delete_positions.extend(range(s, s+len(d)))
            
        cleaned_new_comment_tokens = []
        for i, tok in enumerate(new_comment_tokens):
            if i not in delete_positions:
                cleaned_new_comment_tokens.append(tok)
        
        new_comment_tokens = cleaned_new_comment_tokens
        
    for d, i in post_update:
        valid_positions = get_valid_positions(' '.join(d), ' '.join(new_comment_tokens))
        for v in valid_positions:
            if v + len(i) >= len(new_comment_tokens) or new_comment_tokens[v:v+len(i)] != i:
                new_comment_tokens[v:v+len(d)] = i
    
    return ' '.join(new_comment_tokens)

def format_diff_commands(reference_tokens, commands):
        i = 0
        ref_ptr = 0
        output = []
        
        while i < len(commands):
            command = commands[i]
            if command in [DEL, UPDATEFROM]:
                ref_ptr += 1
            elif command == KEEP:
                if ref_ptr < len(reference_tokens):
                    output.append(reference_tokens[ref_ptr])
                    ref_ptr += 1
            elif command not in [INSERT, UPDATETO]:
                output.append(command)
            i += 1
        return ' '.join(output)

def format_diff_tokens(diff_tokens):
    i = 0
    output = []
    last_command = KEEP

    while i < len(diff_tokens):
        token = diff_tokens[i]
        if token in [INSERT, DEL, UPDATEFROM, UPDATETO, KEEP]:
            last_command = token
        elif last_command in [INSERT, UPDATETO, KEEP]:
            output.append(token)
        i += 1
    return ' '.join(output)

def format_diff_spans(reference_tokens, diff_span_tokens):
    def get_next_keep_token(start_idx, sequence):
            while start_idx < len(sequence) and sequence[start_idx] != KEEP:
                start_idx += 1
            
            start_idx += 1
            if start_idx < len(sequence):
                return sequence[start_idx]
            return None
    
    ptr = 0
    output = reference_tokens.copy()

    i = 0
    while i < len(diff_span_tokens):
        token = diff_span_tokens[i]
        i += 1

        if token not in [INSERT, DEL, UPDATEFROM, KEEP]:
            continue
        
        if token == INSERT:
            j = i

            next_keep_token = get_next_keep_token(j, diff_span_tokens)
            if next_keep_token:
                copy_ptr = ptr
                while copy_ptr < len(output) and output[copy_ptr] != next_keep_token:
                    copy_ptr += 1
                if copy_ptr < len(output):
                    ptr = copy_ptr
            elif ptr < len(output):
                ptr = len(output)
                
            while j < len(diff_span_tokens) and diff_span_tokens[j] != INSERT_END:
                output.insert(ptr, diff_span_tokens[j])
                ptr += 1
                j += 1
            
            i = j+1
        
        elif token == DEL:
            j = i
            while j < len(diff_span_tokens) and diff_span_tokens[j] != DEL_END:
                copy_ptr = max(0, ptr-1)
                while copy_ptr < len(output) and diff_span_tokens[j] != output[copy_ptr]:
                    copy_ptr += 1
                if copy_ptr < len(output):
                    output.pop(copy_ptr)
                    ptr = copy_ptr
                else:
                    ptr += 1
                j += 1
            i = j+1
        
        elif token == KEEP:
            j = i
            while j < len(diff_span_tokens) and diff_span_tokens[j] != KEEP_END:
                if ptr < len(output) and diff_span_tokens[j] == output[ptr]:
                    ptr += 1
                j += 1
            i = j+1
        else:
            j = i
            while j < len(diff_span_tokens) and diff_span_tokens[j] != UPDATETO:
                copy_ptr = max(0, ptr-1)
                while copy_ptr < len(output) and diff_span_tokens[j] != output[copy_ptr]:
                    copy_ptr += 1
                if copy_ptr < len(output):
                    output.pop(copy_ptr)
                    ptr = copy_ptr
                else:
                    ptr += 1
                j += 1
                
            j += 1
            next_keep_token = get_next_keep_token(j, diff_span_tokens)
            if next_keep_token:
                copy_ptr = ptr
                while copy_ptr < len(output) and output[copy_ptr] != next_keep_token:
                    copy_ptr += 1
                if copy_ptr < len(output):
                    ptr = copy_ptr
            elif ptr < len(output):
                ptr = len(output)

            while j < len(diff_span_tokens) and diff_span_tokens[j] != UPDATE_END:
                output.insert(ptr, diff_span_tokens[j])
                ptr += 1
                j += 1
            i = j+1
    return ' '.join(output)

