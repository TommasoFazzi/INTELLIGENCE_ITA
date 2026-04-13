import { NextRequest, NextResponse } from 'next/server';

// Platform is fully public — no access restrictions
export function middleware(_req: NextRequest) {
  return NextResponse.next();
}

export const config = {
  matcher: [],
};
