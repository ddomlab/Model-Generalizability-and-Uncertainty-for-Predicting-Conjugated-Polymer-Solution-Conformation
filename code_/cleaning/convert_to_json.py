import json
from pathlib import Path

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / 'datasets'
JSONS: Path = DATASETS/ 'json_resources'

block_copolymers = {'P3HT-b-P3HTNMe3':['block copolyelectrolytes','m and n is defined'],
                    'P3HT-b-P3HTPy':['block copolyelectrolytes','m and n is defined'],
                    'P3HT-b-P3HTIm': ['block copolyelectrolytes','m and n is defined'],
                    'P3HT-b-P3HTPMe3': ['block copolyelectrolytes','m and n is defined'],
                    'PTh-g-PAU5': ['graft copolymers','m and n is defined'],
                    'PTh-g-PAU17':['graft copolymers','m and n is defined'],
                    'PTh-g-PAU48':['graft copolymers','m and n is defined'],
                    'PMI':['graft copolymers','m and n is defined'],
                    'PF-PANI11112b-PANI11':['triblock copolymer','m and n is defined'],
                    'PDY-132': ['Random copolymer', 'x y z is not defined'],
                    'DTVI1-TVB99': ['Random copolymer','x and y is defined'],
                    'DTVI5-TVB95': ['Random copolymer','x and y is defined'],
                    'DTVI10-TVB90': ['Random copolymer','x and y is defined'],
                    'DTVI25-TVB75': ['Random copolymer','x and y is defined'],
                    'DTVI50-TVB50': ['Random copolymer','x and y is defined'],
                    
                    }


correct_structure_name: dict[str,list[str]] = {'PPFOH': ['PPFOH','PPFOH-L'],
                                                'rr-P3HT': ['rr-P3HT', 'P3DT-d21','P3HT (high Mw)','P3HT_a','P3HT_b','P3HT'],
                                                'PQT-12': ['PQT12','PQT-12'],
                                                'PTHS': ['PTHS1','PTHS2','PTHS3','PTHS'],
                                                'PBTTT-C14' : ['pBTTT-C14','PBTTT-C14','PBTTT_C14_1','PBTTT_C14_2','PBTTT_C14_3', 'PBTTT_C14_4'],
                                                'PBTTT-C16' : ['PBTTT-C16','pBTTTC16',],
                                                'PFO' : ['PFO', 'PF8'],
                                                'DPPDTT' : ['DPPDTT1', 'DPPDTT2', 'DPPDTT3', 'DPPDTT'],
                                                'PII-2T' : ['PII-2T', 'High MW PII-2T'],
                                                'MEH-PPV' : ['MEH-PPV', 'MEH-PPV-100', 'MEH-PPV-30', 'MEH-PPV-70',],
                                                'PFT3' : ['S_PFT', 'PFT3'],
                                                'P(NDI2OD-T2)': ['P(NDI2OD-T2)', 'PNDI-C0', 'NDI-C0', 'NDI-2T-2OD'],
                                                'PCPDTPT-ODD': ['PCPDTPT-ODD', 'PCDTPT-ODD']
                                                }


def save_json(file, direction) ->None:
    with direction.open("w") as f:
        json.dump(file, f, indent=4)

block_cp_dir = JSONS/'block_copolymers.json'
corrected_name_dir = JSONS/'canonicalized_name.json'

save_json(block_copolymers, block_cp_dir)
save_json(correct_structure_name,corrected_name_dir)

print('Done saving jsons')