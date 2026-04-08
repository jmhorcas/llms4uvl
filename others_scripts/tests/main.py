from kb import KnowledgeBase


CSV_FILEPATH = '../resources/kb_uvl/OpenAI_GPT-5.2.csv'


def main() -> None:
    kb = KnowledgeBase()
    kb.load_from_csv(CSV_FILEPATH)

    print(f'Total triplets extracted: {len(kb.entries)}')
    
    # Analysis for seed
    for iteration, seed in sorted(set((e.iteration, e.seed) for e in kb.entries)):
        print(f'\nAnalysis for seed: "{seed}" (iteration {iteration})')
        print(f'  |-Total triplets for "{seed}": {len(kb.get_all_triplets_for_seed(seed))}')
        print(f'  |-Unique triplets for "{seed}": {len(kb.get_deduplicated_triplets(seed))}')
        consistent_triplets, inconsistent_triplets, consistency_index = kb.calculate_strict_consistency(seed)
        print(f'  |-Strict Consistency index for "{seed}": {consistency_index:.4f} ({consistency_index * 100:.2f}%)')
        print(f'    |-Strict Consistent triplets for "{seed}" (appear in all runs): {len(consistent_triplets)}')
        print(f'    |-Strict Inconsistent triplets for "{seed}" (appear only once): {len(inconsistent_triplets)}')
        consistent_triplets, inconsistent_triplets, consistency_index = kb.calculate_semantic_consistency(seed)
        print(f'  |-Semantic Consistency index for "{seed}": {consistency_index:.4f} ({consistency_index * 100:.2f}%)')
        print(f'    |-Semantic Consistent triplets for "{seed}" (appear in all runs): {len(consistent_triplets)}')
        print(f'    |-Semantic Inconsistent triplets for "{seed}" (appear only once): {len(inconsistent_triplets)}')

        # Get candidate concepts for the seed
        candidates = kb.get_candidates(top_n=10, min_frequency=2, seed=seed)
        print(f'  |-Top candidate concepts for "{seed}":')
        for concept, count in candidates:
            print(f'    - {concept} (appears in {count} runs)')

        # Get fuzzy candidate concepts for the seed
        candidates = kb.get_fuzzy_candidates(top_n=10, seed=seed)
        print(f'  |-Top fuzzy candidate concepts for "{seed}":')
        for concept, count in candidates:
            print(f'    - {concept} (appears in {count} runs)')

if __name__ == "__main__":
    main()