## Перевірка черги індексації

Перед розбором інциденту переконайтесь, що воркер узагалі отримує повідомлення. Команди нижче виконуються на вузлі з доступом до брокера.

```bash
rabbitmqctl list_queues name messages consumers | grep kb.reindex
rabbitmqctl list_consumers | grep kb.reindex
rabbitmqctl list_queues name messages | grep kb.reindex.dlq
psql "$POSTGRES_DSN" -c "select count(*) from kb.outbox_events where published_at is null"
psql "$POSTGRES_DSN" -c "select index_state, count(*) from kb.article group by index_state"
psql "$POSTGRES_DSN" -c "select max(now() - created_at) from kb.outbox_events where published_at is null"
journalctl -u webitel-kb-indexer --since '10 min ago' | grep -i 'indexing failed'
journalctl -u webitel-kb-indexer --since '10 min ago' | grep -i 'giving up'
consul kv get service/webitel-kb/leader
systemctl status webitel-kb-indexer --no-pager
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 101"
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 102"
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 103"
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 104"
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 105"
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 106"
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 107"
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 108"
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 109"
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 110"
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 111"
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 112"
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 113"
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 114"
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 115"
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 116"
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 117"
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 118"
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 119"
psql "$POSTGRES_DSN" -c "select count(*) from kb.chunk where version_id = 120"
```

Якщо черга росте, а споживачів нуль, воркер не піднявся або втратив зʼєднання з брокером. Дивіться журнал процесу і стан лідера релею.

## Повторна обробка з черги помилок

Повідомлення з `kb.reindex.dlq` не повертаються автоматично. Спершу усувають причину, потім публікують повідомлення назад в обмінник `kb.reindex` вручну.
