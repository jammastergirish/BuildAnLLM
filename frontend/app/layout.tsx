import "katex/dist/katex.min.css";
import "./globals.css";
import type { ReactNode } from "react";
import { Space_Grotesk, Spectral } from "next/font/google";
import Nav from "../components/Nav";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space",
  display: "swap",
});

const spectral = Spectral({
  subsets: ["latin"],
  weight: ["400", "600", "700"],
  variable: "--font-spectral",
  display: "swap",
});

export const metadata = {
  title: "Transformer Studio",
  description: "Training and inference dashboard",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${spaceGrotesk.variable} ${spectral.variable}`}>
      <body>
        <header className="site-header">
          <div className="brand">
            Transformer Studio
            <span>Training and inference</span>
          </div>
          <Nav />
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
