import React from 'react';
import { create } from 'react-test-renderer';
import { act } from 'react';
import { ScannerHUDView } from '../components/ScannerHUDView';

describe('ScannerHUDView', () => {
  const mockProps = {
    isConnected: true,
  };

  it('renders without crashing when connected', () => {
    let tree: any;
    act(() => {
      tree = create(React.createElement(ScannerHUDView, mockProps));
    });
    expect(tree).toBeDefined();
  });

  it('renders without crashing when disconnected', () => {
    let tree: any;
    act(() => {
      tree = create(
        React.createElement(ScannerHUDView, { ...mockProps, isConnected: false })
      );
    });
    expect(tree).toBeDefined();
  });

  it('serializes to tree with CORE_SCAN_V1.0 title', () => {
    let tree: any;
    act(() => {
      tree = create(React.createElement(ScannerHUDView, mockProps));
    });
    const json = JSON.stringify(tree.toJSON());
    expect(json).toContain('CORE_SCAN_V1.0');
  });

  it('includes green color when connected', () => {
    let tree: any;
    act(() => {
      tree = create(React.createElement(ScannerHUDView, mockProps));
    });
    const json = JSON.stringify(tree.toJSON());
    expect(json).toContain('00FF00');
  });

  it('includes red color when disconnected', () => {
    let tree: any;
    act(() => {
      tree = create(
        React.createElement(ScannerHUDView, { ...mockProps, isConnected: false })
      );
    });
    const json = JSON.stringify(tree.toJSON());
    expect(json).toContain('FF0000');
  });
});