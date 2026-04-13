import { redirect } from 'next/navigation';

// Platform is now fully public — redirect legacy access links to dashboard
export default function AccessPage() {
  redirect('/dashboard');
}
