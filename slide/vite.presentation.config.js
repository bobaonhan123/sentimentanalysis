import { defineConfig } from 'vite';

export default defineConfig({
	server: {
		port: Number(process.env.npm_config_port || 8000),
	},
});
