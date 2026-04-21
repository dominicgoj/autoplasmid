import copy
import argparse
import pandas as pd
from scripts.gibson_design import OVS, design_gibson_overhangs, design_binding_primers
from scripts.plasmid_work import insert_into_vector
from scripts.utils import safe_string, gibson_design_args, MIN_SEQ_LEN_FOR_PRIMER, EXT_PER_BP, PRIMER_START_NUM, PLASMID_START_NUM, REACTION_START_NUM, FWD_RV_TRANSLATE
import os
import datetime


EXPORT_FOLDER_NAME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
EXTRA_SEQUENCE_COLUMNS = ['seq_id', 'seq_name', 'plasmid_type', 'fragment_seq']

def read_csv(path, sep=";"):
    file = pd.read_csv(path, sep=sep).copy()
    return file

def read_excel(path):
    file = pd.read_excel(path).copy()
    return file

def clean_seq_white_spaces(df):
    df['gene_seq'] = df['gene_seq'].str.replace(r'\s+', '', regex=True)
    return df

def calculate_gene_seq_length(table):
    table['gene_seq_len'] = table['gene_seq'].str.len()
    # Extension time as Timedelta (minutes are calculated from bp / 2000 bp/min)
    #table['ext_time'] = table['opt_gene_seq_len'].apply(lambda x: pd.Timedelta(minutes=x/EXT_PER_BP))
    #table = table.sort_values(by=['ext_time'])
    return table

def calculate_ext_time(table):
    table['ext_time'] = table['gene_seq_len'].apply(lambda x: pd.Timedelta(minutes=x/EXT_PER_BP))
    return table


def build_short_gene_insert(insert_seq, mode):
    """Build synthetic insert for short genes using static Gibson overhangs."""
    return insert_seq


def generate_construct_and_primers(full_list, mode='pppt4', forward_only=False, rv_primer=None):
    """
    Generalized function to generate Gibson primers and plasmid constructs.
    
    Args:
        full_list: DataFrame with optimized sequences
        mode: 'pppt4' or 'pppt4a' - determines configuration
    """
    global PRIMER_START_NUM, PLASMID_START_NUM
    print(f"[generate_construct_and_primers] Start mode={mode}, targets={len(full_list)}")
    
    # Configuration for each mode
    
    
    os.makedirs(gibson_design_args['pppt4']['output_dir'], exist_ok=True)
    os.makedirs(gibson_design_args['pppt4a']['output_dir'], exist_ok=True)
    
    cfg = gibson_design_args[mode]
    all_primers = []
    all_plasmids = []
    seq_design_cache = {}
    """
    Seq which are shorter than 125 bp are collected extra and exported as info, that
    there are no primers designed for this
    """
    all_extra_sequences = [] 
    
    
    # Design primers for all sequences
    for i, row in full_list.iterrows():
        seq = f"{cfg['ATG_add']}{row['gene_seq']}"
        seq_id = row['seq_id']
        is_short_gene = len(str(row['gene_seq'])) < MIN_SEQ_LEN_FOR_PRIMER
        print(f"[generate_construct_and_primers] Processing seq_id={seq_id} mode={mode} short_gene={is_short_gene}")
        
        safe_name = safe_string(row['name'])
        
        designed_primers = []
        seq_for_gene_field = seq
        if is_short_gene:
            seq_for_gene_field = build_short_gene_insert(insert_seq=seq, mode=mode)
            seq_to_order = {
                'seq_id': seq_id,
                'seq_name': safe_name,
                'plasmid_type': mode,
                'fragment_seq': f"{OVS[mode]['FWD_OV']}{row['gene_seq']}{OVS[mode]['RV_OV_COMP']}",
                
            }
            all_extra_sequences.append(seq_to_order)
            print(f"[generate_construct_and_primers] Added extra sequence for short gene seq_id={seq_id} mode={mode}")
            
        else:
            primers = design_binding_primers(insert_seq=seq, seq_id=seq_id, mode=mode, forward_only=forward_only)
            if primers is not None:
                for primer in primers:
                    primer['name'] = safe_name
                    primer = design_gibson_overhangs(primer=primer, mode=mode)
                    designed_primers.append(primer)

        # Primers for plasmid map can differ from export list when reusing an existing reverse primer.
        primers_for_map = list(designed_primers)
        if (
            forward_only
            and rv_primer is not None
            and not is_short_gene
            and isinstance(rv_primer, dict)
            and rv_primer.get('dir') == 'rv'
        ):
            reused_rv = copy.deepcopy(rv_primer)
            reused_rv['seq_id'] = row['seq_id']
            reused_rv['gene_seq'] = seq_for_gene_field
            primers_for_map.append(reused_rv)

        seq_design_cache[seq_id] = {
            'is_short_gene': is_short_gene,
            'insert_seq': seq_for_gene_field,
            'primers': designed_primers,
            'primers_for_map': primers_for_map
        }

        if not is_short_gene:
            for primer in designed_primers:
                primer['seq_id'] = row['seq_id']
                primer['gene_seq'] = seq_for_gene_field
                primer['primer_num'] = PRIMER_START_NUM
                primer['primer_name'] = f"P{PRIMER_START_NUM}_F{row['seq_id']}_{primer['dir']}"
                primer['plasmid_type'] = mode
                PRIMER_START_NUM += 1
                all_primers.append(primer)
    
    gibson_primers_df = pd.DataFrame(data=all_primers)
    print(f"[generate_construct_and_primers] Designed primers mode={mode}: {len(gibson_primers_df)}")
    #gibson_primers_df.to_excel(cfg['excel_output'])
    
    # Generate constructs
    for i, target in full_list.iterrows():
        seq = f"{cfg['ATG_add']}{target['gene_seq']}"
        safe_name = safe_string(target['name'])
        seq_id = target['seq_id']
        plasmid_number = PLASMID_START_NUM
        construct_name = f"{cfg['prefix']}_V{plasmid_number}_{safe_name}"
        plasmid_nr = f"V{plasmid_number}"
        design_info = seq_design_cache.get(seq_id, {})
        is_short_gene = bool(design_info.get('is_short_gene', False))
        insert_for_vector = design_info.get('insert_seq', seq)
        features_to_draw = None
        if is_short_gene:
            if mode == 'pppt4a':
                seq_id_for_pppt4a = seq_id+1
            else:
                seq_id_for_pppt4a = seq_id
            features_to_draw = [
                {
                    'start': 0,
                    'end': len(insert_for_vector),
                    'relative_to_insert': True,
                    'type': 'misc_feature',
                    'label': f"{safe_name}_F{seq_id_for_pppt4a}",
                    'qualifiers': {
                        'sequence': insert_for_vector,
                        'note': 'Short gene fragment with Gibson overhangs'
                    }
                }
            ]

        target_primers = None
        if not is_short_gene:
            target_primers_df = pd.DataFrame()
            if not gibson_primers_df.empty:
                target_primers_df = gibson_primers_df[gibson_primers_df['seq_id'] == seq_id].copy()
                target_primers_df['plasmid_number'] = plasmid_number
                target_primers_df['plasmid_number_with_V'] = f"V{plasmid_number}"
                gibson_primers_df.loc[target_primers_df.index, 'plasmid_number'] = plasmid_number
                gibson_primers_df.loc[target_primers_df.index, 'plasmid_number_with_V'] = f"V{plasmid_number}"

            target_primers = target_primers_df.to_dict('records')
            for primer in design_info.get('primers_for_map', []):
                if primer.get('dir') != 'rv':
                    continue
                if primer.get('primer_num') is not None and primer.get('plasmid_type') == mode:
                    continue
                if any(existing.get('primer_name') == primer.get('primer_name') for existing in target_primers):
                    continue
                target_primers.append(primer)

            if not target_primers:
                target_primers = None

        insert_into_vector(
            vector_gb_path=cfg['vector_path'],
            insert_seq=insert_for_vector,
            delete_start=cfg['delete_start'],
            delete_end=cfg['delete_end'],
            output_path=f"{cfg['output_dir']}/{cfg['prefix']}_V{plasmid_number}_{safe_name}.gb",
            construct_name=construct_name,
            insert_name=safe_name,
            primers=target_primers,
            features_to_draw=features_to_draw,
            mode=mode
        )
        print(f"[generate_construct_and_primers] Wrote plasmid map {construct_name}")

        all_plasmids.append({
            'nr': plasmid_nr,
            'name': construct_name,
            'secretion': 0 if mode == 'pppt4' else 1,
            'date_of_trafo': '',
            'sequenced': '',
            'glycerol_stock': '',
            'storage': '',
            'vectormap': ''
        })
        PLASMID_START_NUM += 1

    plasmid_list_df = pd.DataFrame(data=all_plasmids)
    extra_sequences_df = pd.DataFrame(data=all_extra_sequences)
    print(
        f"[generate_construct_and_primers] Done mode={mode}: plasmids={len(plasmid_list_df)}, "
        f"extra_sequences={len(extra_sequences_df)}"
    )
    return gibson_primers_df, plasmid_list_df, extra_sequences_df

def generate_pppt4(full_list):
    """Wrapper for pPpT4 construct generation"""
    gibson_primers_df, plasmid_list_df, all_extra_sequences = generate_construct_and_primers(full_list, mode='pppt4')
    return gibson_primers_df, plasmid_list_df, all_extra_sequences

def generate_pppt4a(full_list, forward_only=False, rv_primer=None):
    """Wrapper for pPpT4a construct generation"""
    gibson_primers_df, plasmid_list_df, all_extra_sequences = generate_construct_and_primers(full_list, mode='pppt4a', forward_only=forward_only, rv_primer=rv_primer)
    return gibson_primers_df, plasmid_list_df, all_extra_sequences

def generate_interleaved_primers(full_list):
    """Generate primers per gene in order: pppt4 first, then pppt4a."""
    print(f"[generate_interleaved_primers] Start interleaved generation for {len(full_list)} targets")
    pppt4_results = []
    pppt4a_results = []
    pppt4_plasmids = []
    pppt4a_plasmids = []
    all_extra_sequences_pppt4 = []
    all_extra_sequences_pppt4a = []

    for idx in full_list.index:
        single_target = full_list.loc[[idx]].copy()
        print(f"[generate_interleaved_primers] Target index={idx}, seq_id={single_target.iloc[0]['seq_id']}")
        pppt4_primers, pppt4_plasmid_list, all_ex_seq_pppt4 = generate_pppt4(single_target)
        reused_rv_primer = None
        if pppt4_primers is not None and not pppt4_primers.empty:
            pppt4_rv = pppt4_primers[pppt4_primers['dir'] == 'rv']
            if not pppt4_rv.empty:
                reused_rv_primer = pppt4_rv.iloc[0].to_dict()

        pppt4a_primers, pppt4a_plasmid_list, all_ex_seq_pppt4a = generate_pppt4a(single_target,
                                                                                 forward_only=True,
                                                                                 rv_primer=reused_rv_primer)
        pppt4_results.append(pppt4_primers)
        pppt4a_results.append(pppt4a_primers)
        pppt4_plasmids.append(pppt4_plasmid_list)
        pppt4a_plasmids.append(pppt4a_plasmid_list)
        all_extra_sequences_pppt4.append(all_ex_seq_pppt4)
        all_extra_sequences_pppt4a.append(all_ex_seq_pppt4a)

    pppt4_primer_list = pd.concat(pppt4_results, ignore_index=True) if pppt4_results else pd.DataFrame()
    pppt4a_primer_list = pd.concat(pppt4a_results, ignore_index=True) if pppt4a_results else pd.DataFrame()
    pppt4_plasmid_list = pd.concat(pppt4_plasmids, ignore_index=True) if pppt4_plasmids else pd.DataFrame()
    pppt4a_plasmid_list = pd.concat(pppt4a_plasmids, ignore_index=True) if pppt4a_plasmids else pd.DataFrame()

    pppt4_non_empty_extra = [df for df in all_extra_sequences_pppt4 if df is not None and not df.empty]
    pppt4a_non_empty_extra = [df for df in all_extra_sequences_pppt4a if df is not None and not df.empty]

    pppt4_all_extra_sequences = (
        pd.concat(pppt4_non_empty_extra, ignore_index=True)
        if pppt4_non_empty_extra else pd.DataFrame(columns=EXTRA_SEQUENCE_COLUMNS)
    )
    pppt4a_all_extra_sequences = (
        pd.concat(pppt4a_non_empty_extra, ignore_index=True)
        if pppt4a_non_empty_extra else pd.DataFrame(columns=EXTRA_SEQUENCE_COLUMNS)
    )
    print(
        "[generate_interleaved_primers] Done: "
        f"pppt4_primers={len(pppt4_primer_list)}, "
        f"pppt4a_primers={len(pppt4a_primer_list)}, "
        f"pppt4_plasmids={len(pppt4_plasmid_list)}, "
        f"pppt4a_plasmids={len(pppt4a_plasmid_list)}"
    )
    return pppt4_primer_list, pppt4a_primer_list, pppt4_plasmid_list, pppt4a_plasmid_list, pppt4_all_extra_sequences, pppt4a_all_extra_sequences


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate plasmid maps, primer list and optional short-gene extra sequence list.'
    )
    parser.add_argument(
        '-i',
        '--input-file',
        default='input/targets_curated.xlsx',
        help='Path to input target table (Excel format).'
    )
    parser.add_argument(
        '-p',
        '--primer-start-num',
        type=int,
        default=0,
        help='Optional. Start number for primer naming/counting (default: 0).'
    )
    parser.add_argument(
        '-v',
        '--plasmid-start-num',
        type=int,
        default=0,
        help='Optional. Start number for plasmid naming/counting (default: 0).'
    )
    return parser.parse_args()

def assign_primer_pairs(full_list, primer_list):
    """
    Assigns primers to full_list based on matching seq_id.
    
    Args:
        full_list: DataFrame with sequences to be assigned primers
        primer_list: DataFrame with primers (must contain primer_name, seq_id, and mt columns)
    
    Returns:
        DataFrame with all columns from full_list + 'primers' column (comma-separated primer names) 
        + 'annealing_temp' column (lower mt of the two primers) + 'plasmid_number'
    """
    result = full_list.copy()
    if primer_list is None or primer_list.empty:
        result['primers'] = ''
        result['annealing_temp'] = pd.NA
        result['plasmid_number'] = pd.NA
        return result
    
    # Group primers by seq_id and aggregate primer names + take min mt + plasmid number
    primer_agg = primer_list.groupby('seq_id').agg({
        'primer_name': lambda x: ', '.join(x),
        'mt': 'min',  # Take the lower melting temperature for annealing
        'plasmid_number': 'first'
    }).reset_index()
    
    primer_agg = primer_agg.rename(columns={'primer_name': 'primers', 'mt': 'annealing_temp'})
    
    # Merge with full_list on seq_id
    result = result.merge(primer_agg, on='seq_id', how='left')
    
    # Fill NaN values with empty string for primers
    result['primers'] = result['primers'].fillna('')
    
    # Round annealing_temp down to nearest integer
    result['annealing_temp'] = result['annealing_temp'].apply(lambda x: int(x) if pd.notna(x) else x)

    
    return result

    
if __name__ == '__main__':
    args = parse_args()

    PRIMER_START_NUM = args.primer_start_num
    PLASMID_START_NUM = args.plasmid_start_num

    print('[main] Starting plasmid cloning pipeline')
    print(f"[main] Input file: {args.input_file}")
    print(f"[main] Primer start number: {PRIMER_START_NUM}")
    print(f"[main] Plasmid start number: {PLASMID_START_NUM}")

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    print('[main] Loading targets')
    targets = pd.read_excel(args.input_file)
    print(f"[main] Loaded {len(targets)} target rows")

    print('[main] Cleaning and annotating targets')
    targets = calculate_gene_seq_length(targets)
    targets = clean_seq_white_spaces(targets)

    print('[main] Generating primers and plasmids')
    pppt4_primer_list, pppt4a_primer_list, pppt4_plasmid_list, pppt4a_plasmid_list, all_ex_seq_pppt4, all_ex_seq_pppt4a = generate_interleaved_primers(targets)

    print('[main] Building export tables')
    primer_list = pd.concat(objs=[pppt4_primer_list, pppt4a_primer_list], axis=0)
    primer_list = primer_list.sort_values(by=['seq_id'])
    plasmid_list = pd.concat(objs=[pppt4_plasmid_list, pppt4a_plasmid_list], axis=0)
    plasmid_list['_plasmid_sort'] = pd.to_numeric(
        plasmid_list['nr'].astype(str).str.extract(r'(\d+)', expand=False),
        errors='coerce'
    )
    
    plasmid_list = plasmid_list.sort_values(by=['_plasmid_sort', 'nr']).drop(columns=['_plasmid_sort']).reset_index(drop=True)
    os.makedirs(f"output/{EXPORT_FOLDER_NAME}/", exist_ok=True)
    print(f"[main] Output folder: output/{EXPORT_FOLDER_NAME}/")
    
    all_ex_seq = pd.DataFrame(columns=EXTRA_SEQUENCE_COLUMNS)
    all_ex_seq_parts = [df for df in [all_ex_seq_pppt4, all_ex_seq_pppt4a] if df is not None and not df.empty]
    if all_ex_seq_parts:
        all_ex_seq = pd.concat(objs=all_ex_seq_parts, axis=0, ignore_index=True)
        all_ex_seq = all_ex_seq.sort_values(by=['seq_id'])

    def _add_plasmid_to_primer_name(name, direction, plasmid):
        if pd.isna(name) or pd.isna(plasmid) or str(plasmid).strip() == '':
            return name
        if direction == 'fwd' and str(name).endswith('_fwd'):
            return f"{str(name)[:-4]}_{plasmid}_fwd"
        if direction == 'rv' and str(name).endswith('_rv'):
            return f"{str(name)[:-3]}_{plasmid}_rv"
        return f"{name}_{plasmid}"
    
    primer_list_export = primer_list[['primer_num', 'primer_name', 'dir', 'full_seq', 'full_length', 'binding_length', 'mt', 'gc', 'plasmid_number_with_V']].copy()
    
    primer_list_export['primer_name'] = primer_list_export.apply(
        lambda row: _add_plasmid_to_primer_name(row['primer_name'], row['dir'], row['plasmid_number_with_V']),
        axis=1
    )
    primer_list_export['dir'] = primer_list_export['dir'].map(FWD_RV_TRANSLATE)
    primer_list_export.columns = ['nr', 'name', 'direction', 'sequence', 'primer length', 'binding length', 'melt', 'gc', 'plasmid']
    print('[main] Writing primer_list.xlsx and plasmid_list.xlsx')
    primer_list_export.to_excel(f"output/{EXPORT_FOLDER_NAME}/primer_list.xlsx", index=False)
    plasmid_list.to_excel(f"output/{EXPORT_FOLDER_NAME}/plasmid_list.xlsx", index=False)
    if not all_ex_seq.empty:
        print('[main] Writing seq_no_primers.xlsx')
        all_ex_seq.to_excel(f"output/{EXPORT_FOLDER_NAME}/seq_no_primers.xlsx", index=False)
    else:
        print("No short genes found; skipping seq_no_primers.xlsx export.")

    print('[main] Pipeline completed successfully')
