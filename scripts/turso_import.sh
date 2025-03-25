DB=$1

sqlite3 $DB .dump logs > alignment_logs.sql
sed -i '' 's/INSERT INTO logs/INSERT OR IGNORE INTO logs/g' alignment_logs.sql
turso db shell counterfactual-evaluation < alignment_logs.sql
rm -rf alignment_logs.sql
