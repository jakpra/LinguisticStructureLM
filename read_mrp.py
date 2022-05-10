import json
import random

import transformers


DUMMY_LABEL = '_DUMMY_'
INV_LABEL = '_INV_'
UNA_LABEL = '_UNA_'

tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
special_tok_len = len('<|endoftext|>')

# TODO: run shuffled-history-anchors ablation to quantify importance of relative token anchoring
# TODO: maybe run tokens-anchors-only and labels-only ablations, but may be obsolete?


def permute_edge_labels(orig_edges, seed=42):
    random.seed(seed)
    labels = []
    unlabeled_edges = []
    for e in orig_edges:
        source_id = e['source']
        target_id = e['target']
        label = e['label']
        unlabeled_edges.append((source_id, target_id))
        labels.append(label)
    random.shuffle(labels)
    permuted_edges = []
    for l, (src, tgt) in zip(labels, unlabeled_edges):
        permuted_edges.append({'source': src, 'target': tgt, 'label': l})
    return permuted_edges


def permute_node_anchors(orig_nodes, keep_una=False, seed=42):
    random.seed(seed)
    anchors = []
    permuted_nodes = []
    orig_unanchored_nodes = []
    for node in orig_nodes:
        if node.get('anchors', []):
            if keep_una:
                anchors.append(sorted(node['anchors'], key=lambda x: x['from']))
            else:
                for anchor in sorted(node['anchors'], key=lambda x: x['from']):
                    anchors.append([anchor])
            unanchored_node = node.copy()
            unanchored_node['anchors'] = []
            permuted_nodes.append(unanchored_node)
        else:
            orig_unanchored_nodes.append(node)
    while anchors:
        random.shuffle(anchors)
        random.shuffle(permuted_nodes)
        for n in permuted_nodes[:len(anchors)]:
            a = anchors.pop()
            n['anchors'].extend(a)
    return sorted(orig_unanchored_nodes + permuted_nodes, key=lambda x: x['id'])


def read_mrp_file(filename, permute_labels=False, permute_anchors=False, keep_una=True, seed=42):
    '''
    Reads the json-based MRP format, slightly enriches it, and moves edge information into node-based parent and
    children dictionaries.
    '''
    result = []
    edge_labels = {}
    with open(filename, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            text_id = d.get('id')
            text = d.get('input')
            tops = d.get('tops', [])  # DM doesn't (always?) have tops
            nodes = d.get('nodes')  # DM might not have nodes
            edges = d.get('edges', [])  # PTG doesn't always have edges
            if permute_labels:
                edges = permute_edge_labels(edges, seed=seed)
            if permute_anchors:
                nodes = permute_node_anchors(nodes, keep_una=keep_una, seed=seed)
            text2node = {}
            node2text = {}
            id2node = {}
            chars2node = {}
            for node in nodes:
                node_id = node.get('id')
                if node_id in tops:
                    node['is_top'] = True
                anchor_text = ''
                first_anchor_start = 0
                last_anchor_end = -1

                # TODO: do LM tokenization here and insert UNA nodes for each non-start-of-word token
                #  (currently in data_loop)

                for i, anchor in enumerate(sorted(node.get('anchors', []), key=lambda x: x['from'])):
                    if i == 0:
                        first_anchor_start = anchor['from']
                    else:
                        dist = anchor['from'] - last_anchor_end
                        if dist >= 1:
                            anchor_text += ' '
                            if dist > 1:
                                anchor_text += '... '
                    anchor_text += text[anchor['from']:anchor['to']]
                    last_anchor_end = anchor['to']
                if anchor_text:
                    text2node[anchor_text] = node_id
                    node2text[node_id] = anchor_text
                    node['offset'] = first_anchor_start
                    if first_anchor_start not in chars2node:
                        chars2node[first_anchor_start] = []
                    chars2node[first_anchor_start].append(node_id)
                node['parents'] = []
                node['children'] = []
                id2node[node_id] = node
            node_ids_to_check = set(id2node.keys())
            for edge in edges:
                source_id = edge['source']
                target_id = edge['target']
                source = id2node[source_id]
                target = id2node[target_id]
                label = edge['label']
                if 'attributes' in edge:
                    atts = dict(zip(edge['attributes'], edge['values']))
                    if atts.get('remote'):
                        pass
                if label not in edge_labels:
                    edge_labels[label] = len(edge_labels)
                source['children'].append((label, target_id))
                target['parents'].append((source_id, label))

                if target_id in node_ids_to_check:
                    node_ids_to_check.remove(target_id)
                if source_id in node_ids_to_check:
                    node_ids_to_check.remove(source_id)

            dummy_top_source = (sorted(tops) + sorted(id2node.keys()))[0]
            for unchecked_id in node_ids_to_check:  # some nodes in EDS are not connected
                id2node[unchecked_id]['parents'].append((dummy_top_source, DUMMY_LABEL))
                id2node[dummy_top_source]['children'].append((DUMMY_LABEL, unchecked_id))
                dummy_edge = {'source': dummy_top_source, 'target': unchecked_id, 'label': DUMMY_LABEL}
                edges.append(dummy_edge)
            if 0 not in chars2node:  # in case graph has no edges (happens in PTG once, might be a mistake)
                anchor_end = min(chars2node.keys())
                anchor_text = text[:anchor_end]
                dummy_id = len(id2node)
                dummy_node = {'id': dummy_id, 'anchors': [{'from': 0, 'to': anchor_end}],
                              'offset': 0,
                              'parents': [(dummy_top_source, DUMMY_LABEL)], 'children': []}
                id2node[dummy_id] = dummy_node
                chars2node[0] = [dummy_id]
                node2text[dummy_id] = anchor_text
                text2node[anchor_text] = dummy_id
                id2node[dummy_top_source]['children'].append((DUMMY_LABEL, dummy_id))
                dummy_edge = {'source': dummy_top_source, 'target': dummy_id, 'label': DUMMY_LABEL}
                edges.append(dummy_edge)

            result.append({'id2node': id2node,
                           'chars2node': chars2node,
                           'id': text_id,
                           'text': text,
                           'node2text': node2text,
                           'text2node': text2node,
                           'edges': edges,
                           'source': d.get('source', '')})

    return result, edge_labels
