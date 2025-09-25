import type { Metadata } from 'next';
import type { ReactNode } from 'react';
import AppShell from '../components/AppShell';
import { ThemeRegistry } from '../components/ThemeRegistry';
import './globals.css';

export const metadata: Metadata = {
    title: 'ImageTalk',
    description: 'Interactive image conversation demo.',
};

const RootLayout = ({ children }: { children: ReactNode }) => {
    return (
        <html lang="en" suppressHydrationWarning>
            <body suppressHydrationWarning>
                <ThemeRegistry>
                    <AppShell>{children}</AppShell>
                </ThemeRegistry>
            </body>
        </html>
    );
};

export default RootLayout;
