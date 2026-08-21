// BUG-14 / MLOPS-08: load the web app's .env (the operating-point vars) into
// process.env for the mini-services. The launcher starts them without exporting
// ALERT_THRESHOLD / MIN_TRAILING_VOL_BPS / CORRELATION_SYMBOLS / CORR_* / etc.,
// so those values were dead config. Importing this module as a side effect
// (`import './env'`) loads them BEFORE the module body reads process.env.
import { existsSync, readFileSync } from 'fs'
import { dirname, join } from 'path'
import { fileURLToPath } from 'url'

// mini-services/binance-stream/env.ts -> ../../.env = flash-crash-watchdog-web/.env
const ENV_FILE = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '.env')

export function loadEnvFile(file: string = ENV_FILE): void {
  if (!existsSync(file)) return
  try {
    for (const line of readFileSync(file, 'utf8').split('\n')) {
      const m = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/)
      if (!m) continue
      const key = m[1]
      if (key in process.env) continue // a real environment variable wins
      let val = m[2].trim()
      if (val.length >= 2 &&
          ((val.startsWith('"') && val.endsWith('"')) ||
           (val.startsWith("'") && val.endsWith("'")))) {
        val = val.slice(1, -1)
      }
      process.env[key] = val
    }
  } catch (e) {
    console.warn('[env] could not load', file, e)
  }
}

loadEnvFile()
