/*
 * react_vision/index.js
 *
 * Entry point for the React Native app.
 * This file registers the root component.
 */

import { AppRegistry } from 'react-native';
import App from './App';
import { name as appName } from './app.json';

AppRegistry.registerComponent(appName, () => App);
