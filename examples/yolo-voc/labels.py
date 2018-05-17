#!/usr/bin/env python
#
#   Copyright EAVISE
#   Example: Transform annotations for VOCdevkit to the brambox pickle format
#

import os
import xml.etree.ElementTree as ET
import brambox.boxes as bbb

DEBUG = True        # Enable some debug prints with extra information
ROOT = 'data'       # Root folder where the VOCdevkit is located

TRAINSET = [
    ('2012', 'train'),
    ('2012', 'val'),
    ('2007', 'train'),
    ('2007', 'val'),
]
TESTSET = [
    ('2007', 'test'),
]


def identify(xml_file):
    root = ET.parse(xml_file).getroot()
    folder = root.find('folder').text
    filename = os.path.splitext(root.find('filename').text)[0]
    return '{folder}/JPEGImages/{filename}'.format(folder=folder, filename=filename)


if __name__ == '__main__':
    print('Getting training annotation filenames')
    train = []
    for (year, img_set) in TRAINSET:
        with open('{ROOT}/VOCdevkit/VOC{year}/ImageSets/Main/{img_set}.txt'.format(ROOT=ROOT, year=year, img_set=img_set), 'r') as f:
            ids = f.read().strip().split()
        train += ['{ROOT}/VOCdevkit/VOC{year}/Annotations/{xml_id}.xml'.format(ROOT=ROOT, year=year, xml_id=xml_id) for xml_id in ids]

    if DEBUG:
        print('\t{length} xml files'.format(length=len(train)))

    print('Parsing training annotation files')
    train_annos = bbb.parse('anno_pascalvoc', train, identify)
    # Remove difficult for training
    for k, annos in train_annos.items():
        for i in range(len(annos) - 1, -1, -1):
            if annos[i].difficult:
                del annos[i]

    print('Generating training annotation file')
    bbb.generate('anno_pickle', train_annos, '{ROOT}/train.pkl'.format(ROOT=ROOT))

    print()

    print('Getting testing annotation filenames')
    test = []
    for (year, img_set) in TESTSET:
        with open('{ROOT}/VOCdevkit/VOC{year}/ImageSets/Main/{img_set}.txt'.format(ROOT=ROOT, year=year, img_set=img_set), 'r') as f:
            ids = f.read().strip().split()
        test += ['{ROOT}/VOCdevkit/VOC{year}/Annotations/{xml_id}.xml'.format(ROOT=ROOT, year=year, xml_id=xml_id) for xml_id in ids]

    if DEBUG:
        print('\t{length} xml files'.format(length=len(train)))

    print('Parsing testing annotation files')
    test_annos = bbb.parse('anno_pascalvoc', test, identify)

    print('Generating testing annotation file')
    bbb.generate('anno_pickle', test_annos, '{ROOT}/test.pkl'.format(ROOT=ROOT))
