#!/usr/bin/env python
#
#   Copyright EAVISE
#   Example: Transform annotations for VOCdevkit to the brambox pickle format
#

import xml.etree.ElementTree as ET
import brambox.boxes as bbb

ROOT = 'data'       # Root folder where the VOCdevkit is located

TRAINSET = [
    ('2018', 'train'),
]

VALIDSET = [
    ('2018', 'val'),
]

TESTSET = [
    ('2018', 'test'),
]


def identify(xml_file):
    root = ET.parse(xml_file).getroot()
    folder = root.find('folder').text
    filename = root.find('filename').text
    return '{folder}/JPEGImages/{filename}'.format(folder=folder, filename=filename)


def process(name, verbose=True):
    config_dict = {
        'train': ('training',   TRAINSET),
        'valid': ('validation', VALIDSET),
        'test':  ('testing',    TESTSET),
    }

    name = name.lower()
    assert name in config_dict
    description, DATASET = config_dict[name]

    if len(DATASET) == 0:
        return

    print('Getting {description} annotation filenames'.format(description=description))
    dataset = []
    for (year, img_set) in TRAINSET:
        filename = '{ROOT}/VOCdevkit/VOC{year}/ImageSets/Main/{img_set}.txt'.format(ROOT=ROOT, year=year, img_set=img_set)
        with open(filename, 'r') as f:
            ids = f.read().strip().split()
        dataset += [
            '{ROOT}/VOCdevkit/VOC{year}/Annotations/{xml_id}.xml'.format(ROOT=ROOT, year=year, xml_id=xml_id)
            for xml_id in ids
        ]

    if verbose:
        print('\t{len} xml files'.format(
            len=len(dataset),
        ))

    print('Parsing {description} annotation files'.format(description=description))
    dataset_annos = bbb.parse('anno_pascalvoc', dataset, identify)

    print('Generating {description} annotation file'.format(description=description))
    bbb.generate('anno_pickle', dataset_annos, '{ROOT}/{name}.pkl'.format(ROOT=ROOT, name=name))

    print()


if __name__ == '__main__':
    process('train')
    process('valid')
    process('test')
