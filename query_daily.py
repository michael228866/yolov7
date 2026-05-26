"""CLI report tool for the counter database.

Examples:
    python query_daily.py                 # today
    python query_daily.py 2026-05-26      # specific day
    python query_daily.py --last 7        # last 7 days summary
    python query_daily.py --sessions      # list all program runs
"""

import argparse
from datetime import date, timedelta

import db


def fmt_int(n):
    return f'{n:>5d}'


def report_day(day: str) -> None:
    with db.connect() as conn:
        cur = conn.execute(
            "SELECT direction, COUNT(*) FROM events "
            "WHERE substr(ts, 1, 10) = ? GROUP BY direction",
            (day,),
        )
        totals = dict(cur.fetchall())
        in_total = totals.get('in', 0)
        out_total = totals.get('out', 0)

        cur = conn.execute(
            "SELECT substr(ts, 12, 2) AS hr, direction, COUNT(*) "
            "FROM events WHERE substr(ts, 1, 10) = ? "
            "GROUP BY hr, direction ORDER BY hr",
            (day,),
        )
        hourly = {}
        for hr, direction, n in cur.fetchall():
            hourly.setdefault(hr, {'in': 0, 'out': 0})[direction] = n

    print(f'=== Daily report: {day} ===')
    print(f'  IN     : {fmt_int(in_total)}')
    print(f'  OUT    : {fmt_int(out_total)}')
    print(f'  TOTAL  : {fmt_int(in_total)}   (IN only)')
    print(f'  NET    : {in_total - out_total:+5d}')
    print()

    if not hourly:
        print('  (no events)')
        return

    print(f'  {"Hour":>5}  {"IN":>5}  {"OUT":>5}  {"Bar (IN)":<30}')
    print(f'  {"-"*5}  {"-"*5}  {"-"*5}  {"-"*30}')
    max_hourly = max(h['in'] for h in hourly.values()) or 1
    for hr in sorted(hourly):
        row = hourly[hr]
        bar_len = int(row['in'] / max_hourly * 30)
        bar = '#' * bar_len
        print(f"  {hr}:00  {fmt_int(row['in'])}  {fmt_int(row['out'])}  {bar}")


def report_last_n(n: int) -> None:
    today = date.today()
    days = [(today - timedelta(days=i)).isoformat() for i in range(n)]
    days.reverse()

    with db.connect() as conn:
        cur = conn.execute(
            "SELECT substr(ts, 1, 10) AS day, direction, COUNT(*) "
            "FROM events WHERE substr(ts, 1, 10) >= ? "
            "GROUP BY day, direction ORDER BY day",
            (days[0],),
        )
        data = {}
        for day, direction, n_count in cur.fetchall():
            data.setdefault(day, {'in': 0, 'out': 0})[direction] = n_count

    print(f'=== Last {n} days ===')
    print(f'  {"Date":<12}  {"IN":>5}  {"OUT":>5}  {"NET":>5}  {"Bar (IN)":<30}')
    print(f'  {"-"*12}  {"-"*5}  {"-"*5}  {"-"*5}  {"-"*30}')

    in_max = max((d['in'] for d in data.values()), default=0) or 1
    grand_in = grand_out = 0
    for day in days:
        row = data.get(day, {'in': 0, 'out': 0})
        bar = '#' * int(row['in'] / in_max * 30)
        print(f"  {day}  {fmt_int(row['in'])}  {fmt_int(row['out'])}  "
              f"{row['in']-row['out']:+5d}  {bar}")
        grand_in += row['in']
        grand_out += row['out']

    print(f'  {"-"*12}  {"-"*5}  {"-"*5}  {"-"*5}')
    print(f'  {"TOTAL":<12}  {fmt_int(grand_in)}  {fmt_int(grand_out)}  '
          f'{grand_in - grand_out:+5d}')


def report_sessions() -> None:
    with db.connect() as conn:
        cur = conn.execute(
            "SELECT s.id, s.started_at, s.ended_at, s.source, s.weights, "
            "       COALESCE(SUM(CASE WHEN e.direction='in'  THEN 1 ELSE 0 END), 0) AS ic, "
            "       COALESCE(SUM(CASE WHEN e.direction='out' THEN 1 ELSE 0 END), 0) AS oc "
            "FROM sessions s LEFT JOIN events e ON s.id = e.session_id "
            "GROUP BY s.id ORDER BY s.started_at DESC LIMIT 50"
        )
        rows = cur.fetchall()

    if not rows:
        print('No sessions recorded.')
        return

    print(f'=== Sessions (latest 50) ===')
    print(f'  {"id":<12}  {"started":<19}  {"ended":<19}  {"src":<10}  {"IN":>4}  {"OUT":>4}')
    print(f'  {"-"*12}  {"-"*19}  {"-"*19}  {"-"*10}  {"-"*4}  {"-"*4}')
    for sid, start, end, src, weights, ic, oc in rows:
        end_str = end if end else '(running)'
        src_str = (src or '')[:10]
        print(f'  {sid:<12}  {start:<19}  {end_str:<19}  {src_str:<10}  {ic:>4}  {oc:>4}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('day', nargs='?', default=None,
                        help="day in YYYY-MM-DD (default: today)")
    parser.add_argument('--last', type=int, metavar='N',
                        help='show last N days summary')
    parser.add_argument('--sessions', action='store_true',
                        help='list recent program-run sessions')
    args = parser.parse_args()

    db.init_db()

    if args.sessions:
        report_sessions()
    elif args.last:
        report_last_n(args.last)
    else:
        day = args.day or date.today().isoformat()
        report_day(day)
