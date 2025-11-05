'use client';

import { useState, useEffect } from 'react';

type Timetable = Record<string, Record<string, string[]>>;

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
  num_slots_per_day?: number; // Add this line
}

export default function Home() {
  const [solution, setSolution] = useState<SolverSolution | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAndSolveExample = async () => {
    setLoading(true);
    setError(null);
    setSolution(null);
    try {
      const exampleRes = await fetch('http://localhost:8000/example', { cache: 'no-store' });
      if (!exampleRes.ok) throw new Error('Failed to load example data');
      const exampleData = await exampleRes.json();

      const solveRes = await fetch('http://localhost:8000/solve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        cache: 'no-store',
        body: JSON.stringify(exampleData),
      });

      if (!solveRes.ok) {
        const errorData = await solveRes.json();
        throw new Error(errorData.detail || 'Failed to get solution');
      }

      const result: SolverSolution = await solveRes.json();
      setSolution(result);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Error fetching or solving example.';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAndSolveExample();
  }, []);

  const renderTimetable = (timetable: Timetable | null | undefined) => {
    if (!timetable) return null;
    const firstEntity = Object.values(timetable)[0] || {};
    const days = Object.keys(firstEntity);
    const entities = Object.keys(timetable);
    const num_slots_per_day = 6; // Assuming 6 slots as per example data

    return (
      <div className="overflow-x-auto">
        {entities.map((entityId) => (
          <div key={entityId} className="mb-8 p-4 border border-gray-200 rounded-lg bg-white shadow-sm">
            <h3 className="text-xl font-semibold mb-4 text-gray-800">{entityId}</h3>
            <div className="grid" style={{ gridTemplateColumns: `80px repeat(${days.length}, 1fr)` }}>
              {/* Header Row: Time Slots and Days */}
              <div className="font-bold text-center p-2 border-b border-r bg-gray-100">Time</div>
              {days.map((day) => (
                <div key={day} className="font-bold text-center p-2 border-b bg-gray-100">{day}</div>
              ))}

              {/* Timetable Grid Rows */}
              {Array.from({ length: num_slots_per_day }).map((_, slotIdx) => (
                <div key={slotIdx} className="contents">
                  <div className="font-medium text-center p-2 border-r bg-gray-50">Slot {slotIdx}</div>
                  {days.map((day) => (
                    <div
                      key={`${entityId}-${day}-${slotIdx}`}
                      className="p-2 border border-gray-200 flex items-center justify-center min-h-[60px] text-xs leading-tight bg-white"
                    >
                      {timetable[entityId]?.[day] && timetable[entityId][day][slotIdx] !== '-' ? (
                        <span className="bg-blue-200 text-blue-800 px-2 py-1 rounded-md">
                          {timetable[entityId][day][slotIdx].split(':').map((part: string, i: number) => (
                            <p key={i}>{part}</p>
                          ))}
                        </span>
                      ) : (
                        ''
                      )}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderWeightsSummary = (penalties?: Penalties) => {
    if (!penalties) return null;
    const entries = Object.entries(penalties).filter(([k]) => k !== 'total_weighted');
    if (!entries.length) return null;
    return (
      <div className="flex flex-wrap gap-2 mb-3">
        {entries.map(([key, val]) => (
          <span key={key} className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded border">
            {key.replaceAll('_', ' ')}: w={Math.round(val.weight)}
          </span>
        ))}
      </div>
    );
  };

  const renderStats = (stats?: Record<string, number>) => {
    if (!stats) return null;
    const entries = Object.entries(stats);
    if (!entries.length) return null;
    return (
      <div className="my-4">
        <h3 className="text-lg font-semibold mb-2 text-gray-700">Solver stats</h3>
        <div className="flex flex-wrap gap-2">
          {entries.map(([key, val]) => (
            <span key={key} className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded border">
              {key.replaceAll('_', ' ')}: {key === 'wall_time_s' ? val.toFixed(2) + 's' : Math.round(val)}
            </span>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <h1 className="text-4xl font-bold text-center mb-8 text-gray-800">Timetabling Solver</h1>

      {loading ? (
        <p className="text-center text-lg text-gray-700">Loading </p>
      ) : error ? (
        <div className="max-w-4xl mx-auto p-6 rounded-lg shadow-md bg-red-100 border border-red-400 text-red-700">
          <p>Error: {error}</p>
        </div>
      ) : solution && (
        <div className="max-w-4xl mx-auto bg-white p-6 rounded-lg shadow-md">
        
          {solution.objective_value !== null && (
            <p className="mb-2"><strong>Objective (penalty):</strong> {solution.objective_value}</p>
          )}

          {renderStats(solution.stats)}

     

          {solution.penalties && (
            <div className="my-4">
              <h3 className="text-lg font-semibold mb-2 text-gray-700">Penalty breakdown</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="bg-gray-100 text-gray-700">
                      <th className="text-left p-2">Category</th>
                      <th className="text-right p-2">Weight</th>
                      <th className="text-right p-2">Count</th>
                      <th className="text-right p-2">Weighted</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(solution.penalties)
                      .filter(([k]) => k !== 'total_weighted')
                      .map(([key, val]) => (
                        <tr key={key} className="border-b">
                          <td className="p-2 capitalize text-gray-800">{key.replaceAll('_', ' ')}</td>
                          <td className="p-2 text-right">{Math.round(val.weight)}</td>
                          <td className="p-2 text-right">{Math.round(val.count)}</td>
                          <td className="p-2 text-right">{Math.round(val.weighted)}</td>
                        </tr>
                      ))}
                    {solution.penalties.total_weighted && (
                      <tr className="bg-gray-50 font-semibold">
                        <td className="p-2 text-gray-800">Total</td>
                        <td className="p-2 text-right">-</td>
                        <td className="p-2 text-right">-</td>
                        <td className="p-2 text-right">{Math.round(solution.penalties.total_weighted.weighted)}</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {(solution.status === 'OPTIMAL' || solution.status === 'FEASIBLE') ? (
            <>
              <h3 className="text-xl font-semibold mt-6 mb-3 text-gray-700">Class Timetables</h3>
              {renderTimetable(solution.class_timetables)}

              <h3 className="text-xl font-semibold mt-6 mb-3 text-gray-700">Teacher Timetables</h3>
              {renderTimetable(solution.teacher_timetables)}
            </>
          ) : (
            <p className="text-red-600">No optimal or feasible solution found.</p>
          )}
        </div>
      )}
    </div>
  );
}
