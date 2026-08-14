import "./globals.css";
export const metadata = { title: "Aura — Clinical Intelligence", description: "Streaming RAG with citations" };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-zinc-50 text-zinc-900 antialiased">{children}</body>
    </html>
  );
}
