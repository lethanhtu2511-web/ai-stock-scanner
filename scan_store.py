def get_latest_scan(db_path=DEFAULT_DB_PATH):
    """
    Chỉ lấy toàn bộ kết quả thuộc đúng lượt quét mới nhất.
    Không lấy bản ghi mới nhất riêng lẻ của từng mã/chiến lược,
    vì cách đó có thể trộn dữ liệu từ nhiều lượt quét.
    """
    init_scan_db(db_path)

    with connect(db_path) as conn:
        latest_batch = conn.execute("""
            SELECT MAX(scanned_at)
            FROM scan_results
        """).fetchone()[0]

        if not latest_batch:
            return []

        rows = conn.execute("""
            SELECT *
            FROM scan_results
            WHERE scanned_at = ?
            ORDER BY
                CASE signal
                    WHEN 'BUY' THEN 5
                    WHEN 'WAIT' THEN 4
                    WHEN 'WATCH' THEN 3
                    WHEN 'SKIP' THEN 2
                    WHEN 'FILTERED' THEN 1
                    ELSE 0
                END DESC,
                COALESCE(score, 0) DESC,
                CASE strategy
                    WHEN 'SWING' THEN 1
                    WHEN 'REVERSAL' THEN 2
                    WHEN 'PULLBACK' THEN 3
                    ELSE 4
                END ASC,
                symbol ASC
        """, (latest_batch,)).fetchall()

    return [
        _format_row(row)
        for row in rows
    ]
