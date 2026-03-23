import sqlite3

conn = sqlite3.connect('database/cat_emotion.db')
cur = conn.cursor()

migrations = [
    "ALTER TABLE user ADD COLUMN credits INTEGER DEFAULT 20",
    "ALTER TABLE user ADD COLUMN plan VARCHAR(20) DEFAULT 'free'",
    "ALTER TABLE user ADD COLUMN plan_expires_at DATETIME",
    "ALTER TABLE prediction ADD COLUMN credits_used INTEGER DEFAULT 1",
]

for sql in migrations:
    try:
        cur.execute(sql)
        print(f"OK: {sql}")
    except Exception as e:
        print(f"SKIP ({e}): {sql}")

conn.commit()
conn.close()
print("Migration complete.")
