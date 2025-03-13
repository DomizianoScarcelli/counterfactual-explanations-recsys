LOCAL_PATH=$1
TURSO_DB=$2

sqlite3 $LOCAL_PATH .dump > dump.sql
echo ".dump generated for $LOCAL_PATH"
turso db shell $TURSO_DB < dump.sql
rm -rf dump.sql
echo "imported $LOCAL_PATH to $TURSO_DB"
