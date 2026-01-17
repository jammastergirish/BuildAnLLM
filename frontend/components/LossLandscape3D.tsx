
import React, { useMemo } from 'react';
import dynamic from 'next/dynamic';

// Plotly needs to be imported dynamically for Next.js to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

type Coord3D = { x: number; y: number; loss: number };

interface LossLandscape3DProps {
  data: Coord3D[];
}

const LossLandscape3D: React.FC<LossLandscape3DProps> = ({ data }) => {
  const plotData = useMemo(() => {
    if (!data || data.length === 0) return [];

    const x = data.map(d => d.x);
    const y = data.map(d => d.y);
    const z = data.map(d => d.loss);

    return [
      {
        type: 'scatter3d',
        mode: 'lines+markers',
        x: x,
        y: y,
        z: z,
        marker: {
          size: 4,
          color: z,
          colorscale: 'Viridis',
          showscale: true,
          opacity: 0.8,
          colorbar: {
            title: 'Loss',
            thickness: 10,
            len: 0.5,
          }
        },
        line: {
          width: 4,
          color: z,
          colorscale: 'Viridis',
        },
        hoverinfo: 'x+y+z',
      }
    ];
  }, [data]);

  const layout = useMemo(() => {
    return {
      autosize: true,
      margin: { l: 0, r: 0, b: 0, t: 0 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      scene: {
        xaxis: { title: 'Random Proj U' },
        yaxis: { title: 'Random Proj V' },
        zaxis: { title: 'Loss' },
      },
      showlegend: false,
    };
  }, []);

  return (
    <div style={{ width: '100%', height: '400px' }}>
      <Plot
        data={plotData as any}
        layout={layout as any}
        useResizeHandler={true}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
};

export default LossLandscape3D;
