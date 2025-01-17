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
        # G = nx.parse_edgelist(, nodetype = int)

        dump = fp.read()
        j = json.loads(dump)
        locs = j.keys()
        objs = []
        for loc in j.values():
            objs += loc['objects']
    return locs, objs, []


loc2loc_templates = ["What location is next to {} in the story?"]

loc2obj_templates = ["What is in {} in the story?", ]
obj2loc_templates = ["What location is {} in the story?", ]

loc2char_templates = ["Who is in {} in the story?", ]
char2loc_templates = ["What location is {} in the story?", ]

conjunctions = ['and', 'or', 'nor']
articles = ["the", 'a', 'an', 'his', 'her', 'their', 'my', 'its', 'those', 'these', 'that', 'this', 'the']
pronouns = [" He ", " She ", " he ", " she "]


class World:
    def __init__(self, locs, objs, relations, args):
        self.graph = nx.Graph()
        self.graph.add_nodes_from(locs, type='location', fillcolor="yellow", style="filled")
        self.graph.add_nodes_from(objs, type='object')
        self.graph.add_edges_from(relations)

        self.locations = {v for v in locs}
        self.objects = {v for v in objs}
        self.edge_labels = {}

        self.args = args

        with open(args.input_text) as f:
            self.input_text = f.read()
         
        # self.model = QA('model/albert-large-squad')
        self.model = QA('/mnt/e/Masters/Thesis/implementations/KG_construction/LLMs/worldGenKG/model/albert-large-squad')

    def is_connected(self):
        return len(list(nx.connected_components(self.graph))) == 1

    def query(self, query, nsamples=10, cutoff=8):
        return self.model.predictTopK(self.input_text, query, nsamples, cutoff)

    def generateNeighbors(self, nsamples=100):
        self.candidates = {}
        for u in self.graph.nodes:
            self.candidates[u] = {}
            if self.graph.nodes[u]['type'] == "location":
                self.candidates[u]['location'] = self.query(random.choice(loc2loc_templates).format(u), nsamples)
                self.candidates[u]['object'] = self.query(random.choice(loc2obj_templates).format(u), nsamples)
                self.candidates[u]['character'] = self.query(random.choice(loc2char_templates).format(u), nsamples)
            if self.graph.nodes[u]['type'] == "object":
                self.candidates[u]['location'] = self.query(random.choice(obj2loc_templates).format(u), nsamples)
            if self.graph.nodes[u]['type'] == "character":
                self.candidates[u]['location'] = self.query(random.choice(char2loc_templates).format(u), nsamples)

    def relatedness(self, u, v, type='location'):

        s = 0
        u2v, probs = self.candidates[u][type]

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

        v2u, probs = self.candidates[v]['location']

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
            print("NO ANSWER FOUND")
            return None, 0

        for pred, prob in zip(preds, probs):
            t = pred.text
            p = prob
            print('> ', t, p)
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
                print(remove)

                self.input_text = self.input_text.replace(remove, '[MASK]').replace('  ', ' ').replace(' .', '.')
                return t, p
            # else:
            # find a more minimal candidate if possible
            # for pred, prob in zip(preds, probs):
            #     if prob > threshold and "MASK" not in pred.text and len(pred.text) > 2 and pred.text in t:
            #         t = pred.text.strip(string.punctuation)
            #         p = prob
            #         self.input_text = self.input_text.replace(t, '[MASK]').replace('  ', ' ').replace(' .', '.')
            #         print(t, p)
            #         return t, p

        return None, 0

    def _generate(self, filename="entities.json"):

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
        print("=" * 20 + "\tcharacters\t" + "=" * 20)
        self.input_text = tmp
        primer = "Who is somebody in the story?"
        cutoff = cutoffs[0]
        t, p = self.extractEntity(primer, threshold=threshold, cutoff=cutoff)
        while t is not None and len(t) > 1:
            if len(chars) > 1:
                cutoff = cutoffs[0]
            chars.append(t)
            t, p = self.extractEntity(primer, threshold=threshold, cutoff=cutoff)

        print("=" * 20 + "\tlocations\t" + "=" * 20)

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

        print("=" * 20 + "\tobjects\t\t" + "=" * 20)

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

        self.graph.add_nodes_from(locs, type='location', fillcolor="yellow", style="filled")
        self.graph.add_nodes_from(chars, type='character', fillcolor="orange", style="filled")
        self.graph.add_nodes_from(objs, type='object', fillcolor="white", style="filled")

        with open(filename, 'w') as f:
            json.dump({'characters':chars, 'locations':locs, 'objects':objs}, f, indent=4, sort_keys=False)
        self.autocomplete()
    
    def generate(self, filename="entities.json"):

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

        print("Interactive mode, context loaded. Ask questions to see outputs.")
        while True:
            prompt = input("> User: ")
            if prompt == "!exit":
                break

            primer = prompt
            for cutoff in cutoffs:
                print(f"Using cutoff: {cutoff}")
                t, p = self.extractEntity(primer, threshold=threshold, cutoff=cutoff)
                while t is not None and len(t) > 1:
                    print(f'> {t}: ({p})')
                    chars.append(t)
                    t, p = self.extractEntity(primer, threshold=threshold, cutoff=cutoff)

    def autocomplete(self):

        self.generateNeighbors(self.args.nsamples)

        print("=" * 20 + "\trelations\t" + "=" * 20)
        while not self.is_connected():
            components = list(nx.connected_components(self.graph))
            best = (-1, next(iter(components[0])), next(iter(components[1])))

            main = components[0]

            loc_done = True
            for c in components[1:]:
                for v in c:
                    if self.graph.nodes[v]['type'] == 'location':
                        loc_done = False

            for u in main:
                if self.graph.nodes[u]['type'] != 'location':
                    continue

                for c in components[1:]:
                    for v in c:
                        if not loc_done and self.graph.nodes[v]['type'] != 'location':
                            continue
                        best = max(best, (self.relatedness(u, v, self.graph.nodes[v]['type']), u, v))

            _, u, v = best

            # attach randomly if empty or specified
            if _ == 0 or self.args.random:
                candidates = []
                for c in components[0]:
                    if self.graph.nodes[c]['type'] == 'location':
                        candidates.append(c)
                u = random.choice(candidates)
            if self.graph.nodes[u]['type'] == 'location' and self.graph.nodes[v]['type'] == 'location':
                type = "connected to"
            else:
                type = "located in"
            print("{} {} {}".format(v, type, u))
            self.graph.add_edge(v, u, label=type)
            self.edge_labels[(v, u)] = type

    def export(self, filename="graph.dot"):
        nx.nx_pydot.write_dot(self.graph, filename)
        nx.write_gml(self.graph, "graph.gml", stringizer=None)

    def draw(self, filename="graph.svg"):
        self.export()

        if self.args.write_sfdp:
            cmd = "sfdp -x -Goverlap=False -Tsvg graph.dot".format(filename)
            returned_value = subprocess.check_output(cmd, shell=True)
            with open(filename, 'wb') as f:
                f.write(returned_value)
            cmd = "inkscape -z -e {}.png {}.svg".format(filename[:-4], filename[:-4])
            returned_value = subprocess.check_output(cmd, shell=True)
        else:
            nx.draw(self.graph, with_labels=True)
            plt.savefig(filename[:-4] + '.png')

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_text', default='input_text.txt')
    parser.add_argument('--input_text', default='/mnt/e/Masters/Thesis/implementations/data/resolved_zelda_botw.txt')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--nsamples', default=10, type=int)
    # parser.add_argument('--cutoffs', default='fairy', type=str)
    parser.add_argument('--cutoffs', default='11.5 15 12', type=str)
    parser.add_argument('--write_sfdp', default=False, type=bool)
    parser.add_argument('--random', default=False, type=bool)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    world = World([], [], [], args)

    split = args.input_text.split('/')
    filename = split[-1].split('.')[:-1][0]

    # if not os.path.exists('./outputs'):
    #     os.makedirs('./outputs')

    # if not os.path.exists('./outputs/plots'):
    #     os.makedirs('./outputs/plots')

    # if not os.path.exists('./outputs/dot'):
    #     os.makedirs('./outputs/dot')

    # if not os.path.exists('./outputs/entities'):
    #     os.makedirs('./outputs/entities')

    world.generate('./outputs/entities/{}.json'.format(filename))
    # world.draw('./outputs/plots/{}.svg'.format(filename))
    # world.export('./outputs/dot/{}.dot'.format(filename))
