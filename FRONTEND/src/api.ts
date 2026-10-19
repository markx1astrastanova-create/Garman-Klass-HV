import { Streamlit } from 'streamlit-component-lib'

const embedded = import.meta.env.MODE === 'streamlit'
type Reply = { id: string; status: number; body: unknown }
let pending: { id: string; resolve: (response: Response) => void } | null = null
let queue = Promise.resolve()

if (embedded) {
  Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, (event) => {
    const reply: Reply | null = (event as CustomEvent).detail.args.response
    if (reply && pending?.id === reply.id) {
      const { resolve } = pending
      pending = null
      resolve(new Response(JSON.stringify(reply.body), {
        status: reply.status,
        headers: { 'content-type': 'application/json' },
      }))
    }
    Streamlit.setFrameHeight(Math.max(window.innerHeight, 800))
  })
  Streamlit.setComponentReady()
}

export function apiFetch(path: string): Promise<Response> {
  if (!embedded) {
    return fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${path}`)
  }
  // Streamlit component values carry one request at a time.
  const response = queue.then(() => new Promise<Response>((resolve) => {
    const id = crypto.randomUUID()
    pending = { id, resolve }
    Streamlit.setComponentValue({ id, path })
  }))
  queue = response.then(() => undefined, () => undefined)
  return response
}
