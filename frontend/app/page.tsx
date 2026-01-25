'use client';

import { useState, useEffect } from 'react';

type Timetable = Record<string, Record<string, string[]>>;

const PENALTY_LABELS: Record<string, string> = {
  unavailability: 'Teacher unavailability',
  unqualified: 'Unqualified specialist assignment',
  double_period_singles: 'Singles within double subjects',
  triple_difficult: 'Difficult subject overload',
  subject_daily_cap: 'Daily subject cap exceeded',
  total_weighted: 'Total penalty',
};

const humanizePenaltyKey = (key: string) =>
  PENALTY_LABELS[key] ?? key.replaceAll('_', ' ');

type PenaltyEntry = {
  weight: number;
  count: number;
  weighted: number;
};

type Penalties = Record<string, PenaltyEntry>;

interface SolverSolution {
  status: string;
  objective_value: number | null;
  class_timetables: Timetable;
  teacher_timetables: Timetable;
  stats: Record<string, number>;
  penalties?: Penalties;
  num_slots_per_day?: number;
  run_config?: {
    seed: number | null;
    time_limit_seconds: number;
  };
}

export default function Home() {
  const [solution, setSolution] = useState<SolverSolution | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [instanceSize, setInstanceSize] =
    useState<'small' | 'medium' | 'large'>('small');
  const [seedInput, setSeedInput] = useState<string>('0');
  const [timeLimit, setTimeLimit] = useState<number>(20);

  const parseOptionalInt = (value: string) => {
    if (!value.trim()) return null;
    const parsed = Number(value);
    return Number.isNaN(parsed) ? null : parsed;
  };

  const fetchAndSolve = async () => {
    setLoading(true);
    setError(null);
    setSolution(null);

    const seed = parseOptionalInt(seedInput);
    try {
      const exampleRes = await fetch(
        `http://localhost:8000/example?size=${instanceSize}`,
        { cache: 'no-store' }
      );

      if (!exampleRes.ok) {
        throw new Error('Beispieldaten konnten nicht geladen werden.');
      }

      const exampleData = await exampleRes.json();

      const params = new URLSearchParams();
      params.set('time_limit', String(timeLimit));
      if (seed !== null) params.set('seed', String(seed));
      if (seed !== null) params.set('seed', String(seed));

      const solveRes = await fetch(
        `http://localhost:8000/solve?${params.toString()}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          cache: 'no-store',
          body: JSON.stringify(exampleData),
        }
      );

      if (!solveRes.ok) {
        let errorMsg = 'Lösung konnte nicht berechnet werden.';
        try {
          const errorData = await solveRes.json();
          if (errorData?.detail) errorMsg = errorData.detail;
        } catch {}
        throw new Error(errorMsg);
      }

      const result: SolverSolution = await solveRes.json();
      setSolution(result);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Fehler beim Laden oder Lösen des Beispiels.'
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAndSolve();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const downloadSolution = () => {
    if (!solution) return;
    const blob = new Blob([JSON.stringify(solution, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `timetable-run-${solution.run_config?.seed ?? 'seedless'}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const renderTimetable = (timetable?: Timetable | null) => {
    if (!timetable) return null;

    const allDays = new Set<string>();
    Object.values(timetable).forEach((dayMap) =>
      Object.keys(dayMap || {}).forEach((d) => allDays.add(d))
    );

    const days = Array.from(allDays);
    const entities = Object.keys(timetable);
    const numSlots = solution?.num_slots_per_day ?? 6;

    return (
      <div className="overflow-x-auto">
        {entities.map((entityId) => (
          <div
            key={entityId}
            className="mb-8 p-4 border rounded-lg bg-white shadow-sm"
          >
            <h3 className="text-xl font-semibold mb-4">{entityId}</h3>

            <div
              className="grid"
              style={{
                gridTemplateColumns: `80px repeat(${days.length}, 1fr)`,
              }}
            >
              <div className="font-bold text-center p-2 border-b border-r bg-gray-100">
                Zeit
              </div>

              {days.map((day) => (
                <div
                  key={day}
                  className="font-bold text-center p-2 border-b bg-gray-100"
                >
                  {day}
                </div>
              ))}

              {Array.from({ length: numSlots }).map((_, slotIdx) => (
                <div key={slotIdx} className="contents">
                  <div className="font-medium text-center p-2 border-r bg-gray-50">
                    Std. {slotIdx + 1}
                  </div>

                  {days.map((day) => {
                    const cell = timetable[entityId]?.[day]?.[slotIdx];
                    const isEmpty = !cell || cell === '-';

                    return (
                      <div
                        key={`${entityId}-${day}-${slotIdx}`}
                        className="p-2 border flex items-center justify-center min-h-[60px] text-xs"
                      >
                        {!isEmpty && (
                          <span className="bg-blue-200 text-blue-800 px-2 py-1 rounded-md">
                            {cell.split(':').map((part, i) => (
                              <p key={i}>{part}</p>
                            ))}
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderStats = (stats?: Record<string, number>) => {
    if (!stats) return null;

    return (
      <div className="my-4">
        <h3 className="text-lg font-semibold mb-2">Solver-Statistiken</h3>
        <div className="flex flex-wrap gap-2">
          {Object.entries(stats).map(([key, val]) => (
            <span
              key={key}
              className="text-xs bg-gray-100 px-2 py-1 rounded border"
            >
              {key.replaceAll('_', ' ')}:{' '}
              {key === 'wall_time_s' ? `${val.toFixed(2)}s` : Math.round(val)}
            </span>
          ))}
        </div>
      </div>
    );
  };

  const renderPenalties = (penalties?: Penalties) => {
    if (!penalties) return null;

    const entries = Object.entries(penalties).filter(
      ([k]) => k !== 'total_weighted'
    );

    return (
      <div className="my-4">
        <h3 className="text-lg font-semibold mb-2">Strafpunktübersicht</h3>

        <table className="min-w-full text-sm">
          <thead>
            <tr className="bg-gray-100">
              <th className="text-left p-2">Kategorie</th>
              <th className="text-right p-2">Gewicht</th>
              <th className="text-right p-2">Anzahl</th>
              <th className="text-right p-2">Gewichtet</th>
            </tr>
          </thead>

          <tbody>
            {entries.map(([key, val]) => (
              <tr key={key} className="border-b">
                <td className="p-2">{humanizePenaltyKey(key)}</td>
                <td className="p-2 text-right">{Math.round(val.weight)}</td>
                <td className="p-2 text-right">{Math.round(val.count)}</td>
                <td className="p-2 text-right">
                  {Math.round(val.weighted)}
                </td>
              </tr>
            ))}

            {penalties.total_weighted && (
              <tr className="bg-gray-50 font-semibold">
                <td className="p-2">Total penalty</td>
                <td />
                <td />
                <td className="p-2 text-right">
                  {Math.round(penalties.total_weighted.weighted)}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <h1 className="text-4xl font-bold text-center mb-8">
        Stundenplan-Löser
      </h1>

      <div className="max-w-4xl mx-auto mb-6 flex flex-col gap-4">
        <div className="flex gap-4">
          <select
            value={instanceSize}
            onChange={(e) =>
              setInstanceSize(e.target.value as 'small' | 'medium' | 'large')
            }
            className="border rounded px-2 py-1"
          >
            <option value="small">small.json</option>
            <option value="medium">medium.json</option>
            <option value="large">large.json</option>
          </select>

          <button
            onClick={fetchAndSolve}
            className="px-3 py-1 bg-blue-600 text-white rounded"
          >
            Neu lösen
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <label className="flex flex-col text-sm">
            Seed
            <input
              type="number"
              value={seedInput}
              onChange={(e) => setSeedInput(e.target.value)}
              placeholder="Optional"
              className="border rounded px-2 py-1"
            />
          </label>

          <label className="flex flex-col text-sm">
            Time limit (s)
            <input
              type="number"
              min={0.1}
              step={0.1}
              value={timeLimit}
              onChange={(e) => {
                const next = Number(e.target.value);
                setTimeLimit(next > 0 ? next : 0.1);
              }}
              className="border rounded px-2 py-1"
            />
          </label>
        </div>
      </div>

      {loading && <p className="text-center">Lade Daten...</p>}
      {error && <p className="text-center text-red-600">{error}</p>}

      {solution && (
        <div className="max-w-4xl mx-auto bg-white p-6 rounded shadow">
          <div className="flex flex-col gap-2 mb-4">
            <div className="flex flex-wrap gap-4 text-sm">
              <span>
                <strong>Slots pro Tag:</strong>{' '}
                {solution.num_slots_per_day ?? '–'}
              </span>
              {solution.run_config && (
                <>
                  <span>
                    <strong>Seed:</strong>{' '}
                    {solution.run_config.seed ?? 'random'}
                  </span>
                  <span>
                    <strong>Zeitlimit:</strong>{' '}
                    {solution.run_config.time_limit_seconds}s
                  </span>
                </>
              )}
            </div>

            <button
              onClick={downloadSolution}
              className="self-start px-3 py-1 bg-green-600 text-white rounded text-sm"
            >
              Download run JSON
            </button>
          </div>

          {solution.objective_value !== null && (
            <p>
              <strong>Zielfunktion:</strong> {solution.objective_value}
            </p>
          )}

          {renderStats(solution.stats)}
          {renderPenalties(solution.penalties)}

          {(solution.status === 'OPTIMAL' ||
            solution.status === 'FEASIBLE') && (
            <>
              <h3 className="text-xl font-semibold mt-6 mb-3">
                Klassenpläne
              </h3>
              {renderTimetable(solution.class_timetables)}

              <h3 className="text-xl font-semibold mt-6 mb-3">
                Lehrkräftepläne
              </h3>
              {renderTimetable(solution.teacher_timetables)}
            </>
          )}
        </div>
      )}
    </div>
  );
}
