correct_structure_name: dict[str,list[str]] = {'PPFOH': ['PPFOH','PPFOH-L', 'PPFOH-H'],
                                 'P3HT': ['rr-P3HT', 'P3DT-d21','P3HT (high Mw)','P3HT_a','P3HT_b','P3HT'],
                                 'PQT-12': ['PQT12','PQT-12'],
                                 'PTHS': ['PTHS1','PTHS2','PTHS3','PTHS'],
                                 'PBTTT-C14' : ['pBTTT-C14','PBTTT-C14','PBTTT_C14_1','PBTTT_C14_2','PBTTT_C14_3', 'PBTTT_C14_4'],
                                 'PBTTT-C16' : ['PBTTT-C16','pBTTTC16',],
                                 'PFO' : ['PFO', 'PF8'],
                                 'DTVI-TVB' : ['DTVI1-TVB99', 'DTVI5-TVB95', 'DTVI10-TVB90', 'DTVI25-TVB75', 'DTVI50-TVB50', 'DTVI-TVB'],
                                 'DPPDTT' : ['DPPDTT1', 'DPPDTT2', 'DPPDTT3', 'DPPDTT'],
                                 'PII-2T' : ['PII-2T', 'High MW PII-2T'],
                                 'MEH-PPV' : ['MEH-PPV', 'MEH-PPV-100', 'MEH-PPV-30', 'MEH-PPV-70',],
                                 'PFO' : ['PFO', 'PF8', 'PFO-d34'],
                                 'PFT3' : ['S_PFT', 'PFT3'],
                                 'P(NDI2OD-T2)': ['P(NDI2OD-T2)', 'PNDI-C0', 'NDI-C0', 'NDI-2T-2OD'],
                                 }



def mapping_from_external(source, to_main):
        working_main= to_main.copy()
        working_structure = source.set_index('Name', inplace=False)

        # Map combined tuples to the main dataset
        combined_series = working_structure.apply(lambda row: tuple(row.values), axis=1)
        mapped_data = working_main['canonical_name'].map(combined_series)
        unpacked_data = list(zip(*mapped_data))
        print('yes')
        # Assign the unpacked data to the corresponding columns in the dataset
        for idx, col in enumerate(working_structure.columns.tolist()):
            working_main[col] = unpacked_data[idx]
        return working_main


fp_data = m_data.copy()
structural_features = structure_raw_data[['Name', 'SMILES']]    
all_poly_name = set(fp_data['name'])
poly_smiles_name = set(structural_features['Name'])
sym_diff2 = all_poly_name-poly_smiles_name
(sym_diff2)
