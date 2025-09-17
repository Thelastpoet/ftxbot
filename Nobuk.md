---

# 📄 Nobuk Technical Product Specification (PRD)

**Project:** Visitor & Access Control SaaS Platform ("Named: Nobuk")
**Tech Stack:**

* **Backend:** Spring Boot (Java, Spring Framework 6 / Boot 3)
* **Database:** PostgreSQL (Dockerized, managed in production)
* **Frontend:** Next.js (Admin Dashboard), React Native (Guards Mobile App)
* **Infrastructure:** Docker Compose (Dev/On-Prem), Kubernetes (SaaS Production)
* **Notifications:** Bulk SMS Provider API (via REST)
* **Deployment Modes:** Multi-tenant SaaS (default) + On-Premise Single-Tenant (optional)

---

## 1. Product Overview

A SaaS visitor and access control platform for estates, gated communities, and SMEs in Kenya. Guards record resident and visitor entries/exits. Each entry generates a **unique exit code**, sent via SMS, which must be verified before exit. This improves security by preventing unauthorized vehicle or pedestrian departures.

The system is **multi-tenant** by design, but can also be deployed **on-premise per estate** if a community requests full data ownership.

---

## 2. Goals

* Deliver a **secure, reliable, compliant SaaS** for visitor management.
* Enable **real-time notifications** to residents and visitors via SMS.
* Provide **enterprise-grade auditability and access control**.
* Offer **multi-tenant SaaS hosting** and **on-premise deployment option**.
* Scale to serve **dozens of estates** and **thousands of daily logs**.

---

## 3. Key Features

### Visitor & Resident Management

* Register residents (name, phone, house/unit).
* Register visitors (name, phone, vehicle details, purpose).
* Maintain searchable records per tenant.

### Check-In / Check-Out Workflow

* **Check-In**:

  * Guard records entry.
  * System generates **unique alphanumeric exit code**.
  * SMS with code sent to visitor/resident.
  * VisitorLog saved (status = `IN`).

* **Check-Out**:

  * Visitor/resident presents code.
  * Guard verifies code via mobile app.
  * If valid → exit logged (`OUT`), SMS confirmation sent.
  * If invalid → denied, attempt logged, admin alerted.

### Notifications

* SMS alerts to residents on visitor arrival.
* SMS confirmations to visitors on entry/exit.
* Configurable per tenant.

### Admin Dashboard (Web, Next.js)

* Manage residents, guards, and visitor records.
* Export logs (CSV, PDF).
* Analytics (visitor trends, guard activity).
* Configure SMS quotas and notifications.

### Guard Mobile App (React Native)

* Cross-platform (Android-first).
* Offline-first: local DB (SQLite), sync when online.
* Minimal UI for check-in/out + code verification.

### Compliance & Security

* **Spring Security**: JWT-based auth + RBAC (Guard, Resident, Tenant Admin, SaaS Super Admin).
* Audit logs for all sensitive actions.
* Data retention policy (e.g., purge logs >12 months old).
* Right-to-access/export/delete resident data.
* Encryption in transit (TLS) and at rest (Postgres).

### SaaS Business Features

* Tenant self-signup & onboarding.
* Subscription plans (SMS quotas, log retention).
* Payment integrations: Paystack.
* SaaS Super Admin portal: manage tenants, usage, billing.

---

## 4. Personas

* **Guard** → Uses mobile app for check-in/out & code verification.
* **Resident** → Receives SMS alerts & codes, approves/denies visitors.
* **Visitor** -> Receives SMS codes on check-in
* **Tenant Admin (Estate Admin)** → Manages residents, guards, reports.
* **SaaS Super Admin** → Oversees tenants, billing, system health.

---

## 5. Success Metrics

* Guard check-in ≤ 30s.
* Code verification ≤ 20s.
* SMS delivery ≥ 98%.
* Zero cross-tenant data leakage.
* Uptime ≥ 99.9%.
* Onboarding new estate ≤ 1 day.

---

# 🏗️ High-Level Architecture

## 1. System Components

* **Spring Boot Backend (Core Services)**

  * Authentication & RBAC (Spring Security + JWT).
  * Multi-tenancy (row-level `tenantId` by default; schema-per-tenant for on-prem if needed).
  * Visitor check-in/out service.
  * Exit code generator & validator.
  * SMS notification service (via REST API to provider).
  * Audit logging.

* **Database (Postgres)**

  * Shared DB (multi-tenant SaaS mode).
  * Dedicated DB (single-tenant on-premise mode).
  * Entities:

    * `Tenant`
    * `Resident`
    * `Guard`
    * `VisitorLog`
    * `AuditLog`
    * `Subscription`
    * `SMSLog`

* **Frontend (Next.js Web App)**

  * Admin dashboard for tenants.
  * SaaS Super Admin portal.
  * API integration with Spring Boot backend.

* **Mobile App (React Native)**

  * Guard check-in/out UI.
  * Offline sync (SQLite → backend).
  * Simple UX optimized for quick workflows.

* **Infrastructure**

  * **Dev & On-Prem:** Docker Compose (Spring Boot, Postgres, Redis optional).
  * **SaaS Production:** Kubernetes (Spring Boot pods, Postgres managed DB, Redis/Kafka optional).

---

## 2. Data Flows

### Check-In Flow

1. Guard enters visitor details.
2. Mobile app → Spring Boot API.
3. Backend generates exit code.
4. VisitorLog saved (`IN`).
5. SMS sent via provider API.
6. Audit log entry created.

### Check-Out Flow

1. Visitor presents exit code.
2. Guard enters code.
3. Backend validates against VisitorLog.
4. If valid → log updated (`OUT`), SMS confirmation sent.
5. If invalid → deny exit, alert tenant admin.

### Tenant & User Management

* Admin dashboard → REST API → Spring Boot services.
* Users saved with `tenantId`.
* Access enforced via Spring Security RBAC + tenancy filter.

### Billing & Subscription

1. Tenant signs up on web portal.
2. Payment via Stripe or M-Pesa.
3. Subscription stored in DB.
4. Usage enforced (SMS quota, log retention).

---

## 3. Deployment & Operations

### SaaS Mode (Multi-Tenant)

* Cloud-hosted (AWS/GCP/Azure, or local provider).
* One Spring Boot service cluster.
* One shared Postgres DB (row-level tenancy).
* Managed queue system (Kafka/RabbitMQ) if SMS volume grows.

### On-Premise Mode (Single-Tenant)

* Delivered as **Docker Compose** package:

  * `spring-boot-service`
  * `postgres-db`
  * (optional) `redis` for caching
* Runs on estate’s own server/VM.
* Updates via new Docker images.

---

## 4. High-Level Diagram

```
                [ Guard Mobile App ]
                        |
                [ Spring Boot API ]
                        |
        -------------------------------------
        |            |           |           |
   [Auth & RBAC] [VisitorLog] [Code Gen] [SMS Service]
        |                         |
     [Postgres DB]             [SMS Provider API]

 [Resident/Visitor] <--- SMS Codes/Alerts ---> [Bulk SMS Provider]

 [Tenant Admin Web] ---> [Next.js Dashboard] ---> [Spring Boot API]

 [SaaS Super Admin] ---> [Next.js Portal] ---> [Spring Boot API]

 Payments ---> [Stripe/M-Pesa] ---> [Subscription Module]
```

---

## 5. Enterprise-Grade Considerations

* **Multi-Tenancy:** row-level (SaaS), DB-per-tenant (on-prem).
* **Audit Logs:** all sensitive events tracked.
* **Data Retention:** scheduled cleanup jobs (Spring Scheduler).
* **Monitoring:** Spring Boot Actuator + Prometheus + Grafana.
* **Error Tracking:** Sentry or similar.
* **Backups:** encrypted Postgres backups daily.

---

✅ With this setup:

* You get **fast SaaS onboarding** (multi-tenant mode).
* You’re ready for **privacy-conscious estates** (on-prem docker package).
* You’re compliant with **Kenya Data Protection Act**.
* You’re building truly **enterprise-grade**, but focused on **estates & SMEs**, not banks.

---

Do you want me to also expand this into a **phased development roadmap** (Phase 1 = Core Check-In/Out, Phase 2 = SaaS features, Phase 3 = On-Prem deployment), so your Java devs know what to prioritize?
