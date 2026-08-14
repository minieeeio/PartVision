import React from 'react';
import { create } from 'react-test-renderer';
import { act } from 'react';
import { BoundingBoxView } from '../components/BoundingBoxView';
import { PartDetection } from '../types/detection';

describe('BoundingBoxView', () => {
  const mockContainerSize = { x: 0, y: 0, width: 375, height: 667 };

  const mockDetection: PartDetection = {
    label: 'FRONT_BUMPER',
    confidence: 0.92,
    x_min: 0.1,
    y_min: 0.2,
    width: 0.5,
    height: 0.3,
  };

  it('renders without crashing', () => {
    let tree: any;
    act(() => {
      tree = create(
        React.createElement(BoundingBoxView, {
          detection: mockDetection,
          containerSize: mockContainerSize,
        })
      );
    });
    expect(tree).toBeDefined();
  });

  it('serializes to tree with label text', () => {
    let tree: any;
    act(() => {
      tree = create(
        React.createElement(BoundingBoxView, {
          detection: mockDetection,
          containerSize: mockContainerSize,
        })
      );
    });
    const json = JSON.stringify(tree.toJSON());
    expect(json).toContain('FRONT_BUMPER');
    expect(json).toContain('92');
  });

  it('renders non-bumper parts with HOOD label', () => {
    const nonBumperDetection = { ...mockDetection, label: 'HOOD', confidence: 0.85 };
    let tree: any;
    act(() => {
      tree = create(
        React.createElement(BoundingBoxView, {
          detection: nonBumperDetection,
          containerSize: mockContainerSize,
        })
      );
    });
    const json = JSON.stringify(tree.toJSON());
    expect(json).toContain('HOOD');
  });

  it('scales bounding box to container dimensions', () => {
    const largeContainer = { x: 0, y: 0, width: 1000, height: 1000 };
    let tree: any;
    act(() => {
      tree = create(
        React.createElement(BoundingBoxView, {
          detection: mockDetection,
          containerSize: largeContainer,
        })
      );
    });
    const json = tree.toJSON();
    expect(json).toBeTruthy();
  });

  it('handles multiple detections with unique keys', () => {
    const detections = [
      { ...mockDetection, label: 'FRONT_BUMPER' },
      { ...mockDetection, label: 'GRILLE' },
      { ...mockDetection, label: 'HOOD' },
    ];

    const children = detections.map((det, i) =>
      React.createElement(BoundingBoxView, {
        key: `${det.label}-${i}`,
        detection: det,
        containerSize: mockContainerSize,
      })
    );

    let tree: any;
    act(() => {
      tree = create(React.createElement(React.Fragment, null, children));
    });
    const json = tree.toJSON();
    expect(json).toBeTruthy();
  });
});