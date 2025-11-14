// Simple dataset model bridge to vendor MockData
// Usage: import { getDataset, setDataset, regionMap } from '../models/dataset.js'

const hasVendor = typeof window !== 'undefined' && window.MockData;

export function getDataset() {
  if (hasVendor) return window.MockData.getDataset();
  console.warn('MockData vendor not found. Returning empty dataset.');
  return { sales: [], inventory: {}, products: [] };
}

export function setDataset(ds) {
  if (hasVendor) return window.MockData.setDataset(ds);
  console.warn('MockData vendor not found. Skipping setDataset.');
}

export const regionMap = hasVendor ? window.MockData.regionMap : {
  'Hà Nội': 'HN',
  'Đà Nẵng': 'DN',
  'Hồ Chí Minh': 'HCM'
};
