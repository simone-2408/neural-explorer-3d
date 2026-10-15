import './styles/main.css';
import { App } from './app/App';

const root = document.getElementById('app');
if (!root) throw new Error('Missing #app element');
new App(root);
