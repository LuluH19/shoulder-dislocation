// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import mermaid from 'astro-mermaid';

// https://astro.build/config
export default defineConfig({
	integrations: [
		mermaid({ theme: 'neutral', autoTheme: true }),
		starlight({
			title: 'Reconnaissance de mouvement',
			defaultLocale: 'root',
			locales: {
				root: { label: 'Français', lang: 'fr' },
			},
			sidebar: [
				{ label: 'Accueil', link: '/' },
				{
					label: 'Comprendre le projet',
					items: [
						{ label: '1. Vue d’ensemble', slug: 'projet/vue-ensemble' },
						{ label: '2. Les données', slug: 'projet/donnees' },
						{ label: '3. Préparer les données', slug: 'projet/preparation' },
						{ label: '4. Le modèle (CNN)', slug: 'projet/modele' },
						{ label: '5. Entraînement et résultats', slug: 'projet/resultats' },
						{ label: '6. Un modèle pour tout le monde ?', slug: 'projet/generalisation' },
						{ label: '7. Sur la carte (ESP32)', slug: 'projet/embarque' },
					],
				},
				{ label: 'Lexique', slug: 'lexique' },
			],
		}),
	],
});
