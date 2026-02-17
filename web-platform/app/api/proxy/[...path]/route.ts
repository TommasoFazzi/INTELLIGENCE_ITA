import { NextRequest, NextResponse } from 'next/server';

const API_URL = process.env.INTELLIGENCE_API_URL || 'http://localhost:8000';
const API_KEY = process.env.INTELLIGENCE_API_KEY || '';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  const pathStr = path.join('/');
  const searchParams = request.nextUrl.searchParams.toString();
  const upstream = `${API_URL}/api/v1/${pathStr}${searchParams ? `?${searchParams}` : ''}`;

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000); // 300s timeout

    const response = await fetch(upstream, {
      headers: {
        'Content-Type': 'application/json',
        ...(API_KEY && { 'X-API-Key': API_KEY }),
      },
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      return NextResponse.json(
        { success: false, detail: 'Backend request timeout' },
        { status: 504 }
      );
    }
    return NextResponse.json(
      { success: false, detail: 'Backend unavailable' },
      { status: 502 }
    );
  }
}
