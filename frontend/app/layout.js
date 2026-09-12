import "./globals.css";

export const metadata = {
  title: "Football Intelligence Analytics",
  description: "International football match prediction"
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
