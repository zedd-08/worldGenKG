from bert import QA

# !/usr/bin/env python3

import json
import os
import numpy as np
import tensorflow as tf
import matplotlib
import subprocess

matplotlib.use('Agg')

import networkx as nx
import json
import argparse
import matplotlib.pyplot as plt
import random
import string


def readGraph(file):
    with open(file, 'r') as fp:
        dump = fp.read()
        j = json.loads(dump)
        locs = j.keys()
        objs = []
        for loc in j.values():
            objs += loc['objects']
    return locs, objs, []


loc2loc_templates = ["What location is next to {} in the story?"]
loc2obj_templates = ["What is in {} in the story?", ]
loc2char_templates = ["Who is in {} in the story?", ]

obj2loc_templates = ["What location is {} in the story?", ]
obj2char_templates = ["Who has {} in the story?", ]

char2loc_templates = ["What location is {} in the story?", ]

conjunctions = ['and', 'or', 'nor']
articles = ["the", 'a', 'an', 'his', 'her', 'their', 'my', 'its', 'those', 'these', 'that', 'this', 'the']
pronouns = ["He", "She", "he", "she"]


class World:
    def __init__(self, args):
        self.edge_labels = {}

        self.model = QA('model/albert-large-squad')
        self.graph = nx.Graph()
        self.args = args

        if self.args.cutoffs == 'fairy':
            self.cutoffs = [6.5, -7, -5]  # fairy
        elif self.args.cutoffs == 'mystery':
            self.cutoffs = [3.5, -7.5, -6]  # mystery
        else:
            self.cutoffs = [float(i) for i in self.args.cutoffs.split()]
            assert len(self.cutoffs) == 3
    
    def load_from_graph(self, graph):
        self.graph.clear()
        self.graph = graph

    def is_connected(self):
        return len(list(nx.connected_components(self.graph))) == 1

    def query(self, query, nsamples=10, cutoff=8):
        return self.model.predictTopK(self.input_text, query, nsamples, cutoff)

    def generateNeighbors(self, nsamples=100):
        self.candidates = {}
        for u in self.graph.nodes:
            self.candidates[u] = {}
            if self.graph.nodes[u]['type'] == "location":
                self.candidates[u]['location'] = self.query(random.choice(loc2loc_templates).format(u), nsamples, self.cutoffs[1])
                self.candidates[u]['object'] = self.query(random.choice(loc2obj_templates).format(u), nsamples, self.cutoffs[2])
                self.candidates[u]['character'] = self.query(random.choice(loc2char_templates).format(u), nsamples,self.cutoffs[0])
            if self.graph.nodes[u]['type'] == "object":
                self.candidates[u]['location'] = self.query(random.choice(obj2loc_templates).format(u), nsamples, self.cutoffs[1])
                self.candidates[u]['character'] = self.query(random.choice(obj2char_templates).format(u), nsamples, self.cutoffs[0])
            if self.graph.nodes[u]['type'] == "character":
                self.candidates[u]['location'] = self.query(random.choice(char2loc_templates).format(u), nsamples, self.cutoffs[1])

    def relatedness(self, u, v, u_type='location', v_type='location'):
        s = 0
        u2v = None
        v2u = None

        if v_type in self.candidates[u]:
            u2v, probs = self.candidates[u][v_type]

        if u2v is not None:
            for c, p in zip(u2v, probs):
                a = set(c.text.split()).difference(articles)
                b = set(v.split()).difference(articles)

                # find best intersect
                best_intersect = 0
                for x in self.graph.nodes:
                    xx = set(x.split()).difference(articles)
                    best_intersect = max(best_intersect, len(a.intersection(xx)))

                # increment if answer is best match BoW
                if len(a.intersection(b)) == best_intersect:
                    s += len(a.intersection(b)) * p

                # naive method
                # s += len(a.intersection(b)) * p

        if u_type in self.candidates[v]:
            v2u, probs = self.candidates[v][u_type]

        if v2u is not None:
            for c, p in zip(v2u, probs):
                a = set(c.text.split()).difference(articles)
                b = set(u.split()).difference(articles)

                # find best intersect
                best_intersect = 0
                for x in self.graph.nodes:
                    xx = set(x.split()).difference(articles)
                    best_intersect = max(best_intersect, len(a.intersection(xx)))

                # increment if answer is best match BoW
                if len(a.intersection(b)) == best_intersect:
                    s += len(a.intersection(b)) * p

                # naive method
                # s += len(a.intersection(b)) * p

        return s

    def extractEntity(self, query, threshold=0.05, cutoff=0):
        preds, probs = self.query(query, self.args.nsamples, cutoff)

        if preds is None:
            return None, 0

        for pred, prob in zip(preds, probs):
            t = pred.text
            p = prob
            if len(t) < 1:
                continue
            if p > threshold and "MASK" not in t:

                # find a more minimal candidate if possible
                for pred, prob in zip(preds, probs):
                    if t != pred.text and pred.text in t and prob > threshold and len(pred.text) > 2:
                        t = pred.text
                        p = prob
                        break

                t = t.strip(string.punctuation)
                remove = t

                # take out leading articles for cleaning
                words = t.split()
                if words[0].lower() in articles:
                    remove = " ".join(words[1:])
                    words[0] = words[0].lower()
                    t = " ".join(words[1:])

                self.input_text = self.input_text.replace(remove, '[MASK]').replace('  ', ' ').replace(' .', '.')
                return t, p
        return None, 0

    def generate(self):

        locs = []
        objs = []
        chars = []

        # set thresholds/cutoffs
        threshold = 0.05

        if self.args.cutoffs == 'fairy':
            cutoffs = [6.5, -7, -5]  # fairy
        elif self.args.cutoffs == 'mystery':
            cutoffs = [3.5, -7.5, -6]  # mystery
        else:
            cutoffs = [float(i) for i in self.args.cutoffs.split()]
            assert len(cutoffs) == 3

        # save input text
        tmp = self.input_text[:]

        # add chars
        self.input_text = tmp
        primer = "Who is somebody in the story?"
        cutoff = cutoffs[0]
        t, p = self.extractEntity(primer, threshold=threshold, cutoff=cutoff)
        while t is not None and len(t) > 1:
            if len(chars) > 1:
                cutoff = cutoffs[0]
            chars.append(t)
            t, p = self.extractEntity(primer, threshold=threshold, cutoff=cutoff)

        # add locations
        self.input_text = tmp
        primer = "Where is the location in the story?"
        cutoff = cutoffs[1]
        t, p = self.extractEntity(primer, threshold=threshold, cutoff=cutoff)
        while t is not None and len(t) > 1:
            locs.append(t)

            if len(locs) > 1:
                cutoff = cutoffs[1]

            t, p = self.extractEntity(primer, threshold=threshold, cutoff=cutoff)

        # add objects
        self.input_text = tmp
        primer = "What is an object in the story?"
        cutoff = cutoffs[2]
        t, p = self.extractEntity(primer, threshold=threshold, cutoff=cutoff)
        while t is not None and len(t) > 1:
            if len(objs) > 1:
                cutoff = cutoffs[2]
            objs.append(t)
            t, p = self.extractEntity(primer, threshold=threshold, cutoff=cutoff)
        self.input_text = tmp
        
        entities = [(x, 'object') for x in objs]
        entities.extend([(x, 'character') for x in chars])
        entities.extend([(x, 'location') for x in locs])
        return entities

    def autocomplete(self, game_id, id_start):
        self.generateNeighbors(self.args.nsamples)

        while not self.is_connected():
            components = list(nx.connected_components(self.graph))
            best = (-1, next(iter(components[0])), next(iter(components[1])))

            main = components[0]

            for u in main:
                u_type = self.graph.nodes[u]['type']
                for c in components[1:]:
                    for v in c:
                        v_type = self.graph.nodes[v]['type']
                        if u_type != 'location' and u_type == v_type:
                            continue
                        uvrel = self.relatedness(u, v, u_type, v_type)
                        best = max(best, (uvrel, u, v))
            _, u, v = best

            # attach randomly if empty or specified
            if _ == 0 or self.args.random:
                candidates = []
                for c in components[0]:
                    if self.graph.nodes[c]['type'] == 'location':
                        candidates.append(c)
                if len(candidates) == 0:
                    break
                u = random.choice(candidates)
            u_type = self.graph.nodes[u]['type']
            v_type = self.graph.nodes[v]['type']
            
            if u_type == v_type:
                if u_type == 'location':
                    rel_type = "connected_to"
                else:
                    rel_type = "NA" 
            elif u_type == 'location' and v_type in ['object', 'character'] or v_type == 'location' and u_type in ['object', 'character']:
                rel_type = "present_in"
            else:
                rel_type = "held_by"
            
            self.graph.add_edge(v, u, label=rel_type, id=f'{game_id}_E{id_start}')
            id_start += 1
            self.edge_labels[(v, u)] = rel_type
        return id_start