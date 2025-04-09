-- Normalize targets
UPDATE logs
SET gen_target_y_at_1 = '{' || gen_target_y_at_1 || '}'
WHERE gen_strategy = 'targeted_uncategorized'
AND gen_target_y_at_1 NOT LIKE '{%}'  -- Ensures we only update plain numbers.
AND gen_target_y_at_1 GLOB '[0-9]*';  -- Ensures it's a number.

-- Dedupe
DELETE FROM logs
WHERE ROWID NOT IN (
    SELECT MIN(ROWID)
    FROM logs
    GROUP BY 
        CAST(i AS TEXT), 
        CAST(gen_strategy AS TEXT), 
        CAST(split AS TEXT), 
        CAST(model AS TEXT), 
        CAST(dataset AS TEXT), 
        CAST(gen_target_y_at_1 AS TEXT), 
        CAST(pop_size AS TEXT), 
        CAST(generations AS TEXT), 
        CAST(fitness_alpha AS TEXT), 
        CAST(include_sink AS TEXT), 
        CAST(mut_prob AS TEXT), 
        CAST(crossover_prob AS TEXT), 
        CAST(genetic_topk AS TEXT), 
        CAST(mutation_params AS TEXT), 
        CAST(ignore_genetic_split AS TEXT), 
        CAST(jaccard_threshold AS TEXT)
);

-- Remove useless stuff
DELETE FROM logs
WHERE NOT ( 
    CAST(split AS TEXT) = '(None, 10, 0)' 
    AND CAST(determinism AS TEXT) = 'True' 
    AND CAST(generations AS TEXT) = '10' 
    AND CAST(halloffame_ratio AS TEXT) = '0' 
    AND CAST(jaccard_threshold AS TEXT) = '0.5' 
    AND CAST(fitness_alpha AS TEXT) = '0.5' 
    AND CAST(mut_prob AS TEXT) = '0.5' 
    AND CAST(pop_size AS TEXT) = '8192' 
    AND CAST(crossover_prob AS TEXT) = '0.7' 
    AND CAST(genetic_topk AS TEXT) = '1' 
    AND CAST(ignore_genetic_split AS TEXT) = 'True' 
);

