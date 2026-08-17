import "./globals.css";
import { IBM_Plex_Mono, Spectral } from "next/font/google";

/** Prose face. Clinical guidance is read, not scanned, so the model's authored
 *  text is set in a reading serif. */
const prose = Spectral({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  style: ["normal", "italic"],
  variable: "--font-prose",
  display: "swap",
});

/** System face. Everything the system can verify -- source titles, page numbers,
 *  citation markers, retrieval mode -- is monospaced, so at a glance you can tell
 *  what was written from what was proven. */
const system = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-system",
  display: "swap",
});

export const metadata = {
  title: "Aura — Clinical Intelligence",
  description: "Answers from indexed clinical sources, with every claim traceable to its page.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${prose.variable} ${system.variable}`}>
      <body className="antialiased">{children}</body>
    </html>
  );
}
