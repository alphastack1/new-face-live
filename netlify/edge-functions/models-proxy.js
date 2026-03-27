/**
 * Edge function to proxy model downloads from GitHub Releases.
 * GitHub Releases use 302 redirects to release-assets.githubusercontent.com
 * which doesn't set CORS headers. This edge function follows the redirect
 * server-side and streams the response back with proper CORS headers.
 */
export default async (request) => {
  const url = new URL(request.url);
  const filename = url.pathname.replace('/models-cdn/', '');

  if (!filename) {
    return new Response('Not found', { status: 404 });
  }

  const targetUrl = `https://github.com/alphastack1/storage/releases/download/newface-v1/${filename}`;

  try {
    const upstream = await fetch(targetUrl, { redirect: 'follow' });

    if (!upstream.ok) {
      return new Response(`Upstream error: ${upstream.status}`, { status: upstream.status });
    }

    return new Response(upstream.body, {
      status: 200,
      headers: {
        'Content-Type': upstream.headers.get('Content-Type') || 'application/octet-stream',
        'Content-Length': upstream.headers.get('Content-Length') || '',
        'Access-Control-Allow-Origin': '*',
        'Cache-Control': 'public, max-age=31536000, immutable',
      },
    });
  } catch (err) {
    return new Response(`Proxy error: ${err.message}`, { status: 502 });
  }
};

export const config = {
  path: '/models-cdn/*',
};
